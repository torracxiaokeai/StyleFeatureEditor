import os
import sys
import json
import wandb
import datetime
import omegaconf

import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from tqdm.auto import tqdm
from io import BytesIO
from PIL import Image
from time import time
from pathlib import Path
from abc import abstractmethod

from runners.base_runner import BaseRunner
from utils.class_registry import ClassRegistry
from datasets.transforms import transforms_registry
from datasets.datasets import ImageDataset
from datasets.loaders import InfiniteLoader
from training.losses import disc_losses, LossBuilder
from training.optimizers import optimizers
from metrics.metrics import metrics_registry

from training.loggers import Timer, StreamingMeans, TrainigLogger
from utils.common_utils import tensor2im, get_keys
from models.methods import methods_registry

from models.psp.encoders.psp_encoders import ProgressiveStage
from utils.model_utils import toogle_grad


training_runners = ClassRegistry()
        

FACE_DIRECTIONS = {
    "age": [-7, -5, 5, 7, 10],
    "fs_makeup": [5, 8, 12],
    "afro": [0.03, 0.07],
    "angry": [0.06, 0.1],
    "purple_hair": [0.07, 0.1, 0.12],
    "glasses": [-10, -7],
    "face_roundness": [-13, -7, 7, 13], 
    "rotation": [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
    "bobcut": [0.07, 0.12, 0.18],
    "bowlcut": [0.07, 0.14],
    "mohawk": [0.07, 0.10],
    "blond hair": [-8, -4, 4, 8],
    "fs_smiling": [-6, -3, 3, 6, 9]
}


def get_random_edit():
    direction = np.random.choice(list(FACE_DIRECTIONS.keys()))
    strenght = np.random.choice(FACE_DIRECTIONS[direction])
    return direction, strenght
        

@training_runners.add_to_registry(name="base_training_runner")
class BaseTrainingRunner(BaseRunner):
    def setup(self):
        self.start_step = self.config.train.start_step
        self._setup_device()
        self._setup_experiment_dir()

        self._setup_method()
        self._setup_logger()

        self._setup_metrics()
        self._setup_datasets()

        start_batch_size = (
            self.config.train.bs_used_before_adv_loss
            if self.config.train.train_dis
            else self.config.model.batch_size
        )

        self._setup_dataloaders(start_batch_size)

        self._setup_latent_editor()
        self._setup_optimizers()
        self._setup_loss()

    def get_base_model(self):
        """
        获取未被DDP包装的原始模型。
        在分布式训练中，模型会被DistributedDataParallel包装，
        原始模型的属性会存储在.module属性中。
        """
        if self.config.dist.enabled and hasattr(self.method, 'module'):
            return self.method.module
        return self.method
        
    def get_model_component(self, component_name):
        """
        获取模型的组件/属性，自动处理DDP包装的情况。
        
        Args:
            component_name: 要获取的组件名称，如'encoder'、'decoder'等
            
        Returns:
            模型的对应组件
        """
        base_model = self.get_base_model()
        return getattr(base_model, component_name)

    def _setup_logger(self):
        self.logger = TrainigLogger(self.config)

    def _setup_datasets(self):
        print("Loading dataset")
        transform_dict = transforms_registry[self.config.data.transform]().get_transforms()
        self.train_dataset = ImageDataset(
            self.config.data.input_train_dir, transform_dict["train"]
        )

        self.test_dataset = ImageDataset(
            self.config.data.input_val_dir, transform_dict["test"]
        )
        self.paths = self.test_dataset.paths

        self.special_dataset = ImageDataset(
            self.config.data.special_dir, transform_dict["test"]
        )
        self.special_paths = self.special_dataset.paths

    def _setup_dataloaders(self, batch_size):
        # 设置分布式采样器
        if self.config.dist.enabled:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=self.config.dist.world_size,
                rank=self.config.dist.rank,
                shuffle=True
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset,
                num_replicas=self.config.dist.world_size,
                rank=self.config.dist.rank,
                shuffle=False
            )
            special_sampler = torch.utils.data.distributed.DistributedSampler(
                self.special_dataset,
                num_replicas=self.config.dist.world_size,
                rank=self.config.dist.rank,
                shuffle=False
            )
            
            # 分布式环境下，每个进程的batch_size需要缩小
            actual_batch_size = batch_size // self.config.dist.world_size
            if self.config.dist.rank == 0:
                print(f"Adjusting batch size from {batch_size} to {actual_batch_size} per GPU")
        else:
            train_sampler = None
            test_sampler = None
            special_sampler = None
            actual_batch_size = batch_size
        
        # 修改为使用分布式采样器
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=actual_batch_size,
            shuffle=(train_sampler is None),  # 如果使用采样器，不需要shuffle
            sampler=train_sampler,
            num_workers=self.config.model.workers,
            drop_last=True,
            is_infinite=True
        )
        
        # 类似地修改test和special数据加载器
        self.test_dataloader = InfiniteLoader(
            self.test_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=self.config.model.workers,
            is_infinite=False
        )
        
        self.special_dataloader = InfiniteLoader(
            self.special_dataset,
            batch_size=actual_batch_size,
            shuffle=False,
            sampler=special_sampler,
            num_workers=self.config.model.workers,
            is_infinite=False
        )
        
        # 保存采样器引用，以便在epoch开始时设置
        self.train_sampler = train_sampler

    def _setup_optimizers(self):
        # 获取编码器参数，处理DDP情况
        encoder = self.get_model_component('encoder')
        params = list(encoder.parameters())

        optimizer_args = dict(
            self.config.optimizers[self.config.train.encoder_optimizer]
        )
        optimizer_args["params"] = params
        self.encoder_optimizer = optimizers[self.config.train.encoder_optimizer](
            **optimizer_args
        )

        if self.config.model.checkpoint_path != "":
            ckpt = torch.load(self.config.model.checkpoint_path, map_location="cpu")
            if "encoder_opt" in ckpt.keys():
                self.encoder_optimizer.load_state_dict(ckpt["encoder_opt"])
            else:
                print('WARNING, continuing training without loading encoder optimizer state!')

        if self.config.train.train_dis:
            # 获取判别器参数，处理DDP情况
            discriminator = self.get_model_component('discriminator')
            params = list(discriminator.parameters())
            optimizer_args = dict(
                self.config.optimizers[self.config.train.disc_optimizer]
            )
            optimizer_args["params"] = params
            self.disc_optimizer = optimizers[self.config.train.disc_optimizer](
                **optimizer_args
            )

            if self.config.model.checkpoint_path != "":
                if "disc_opt" in ckpt.keys():
                    self.disc_optimizer.load_state_dict(ckpt["disc_opt"])
                else:
                    print('WARNING, continuing training without loading disc optimizer state!')

    def _setup_loss(self):
        enc_losses_dict = self.config.encoder_losses
        disc_losses_dict = self.config.disc_losses

        self.loss_builder = LossBuilder(
            enc_losses_dict, 
            disc_losses_dict, 
            self.device
        )

    def _setup_experiment_dir(self):
        base_root = Path(__file__).resolve().parent.parent
        num = 0
        exp_dir = self.config.exp.exp_dir
        exp_dir_name = "{}_{}".format(self.config.exp.name, str(num).zfill(3))

        # 只有主进程或非分布式环境创建目录
        is_main_process = not self.config.dist.enabled or self.config.dist.rank == 0
        
        if is_main_process:
            exp_path = base_root / exp_dir / exp_dir_name
            while True:
                if exp_path.exists():
                    num += 1
                    exp_dir_name = "{}_{}".format(self.config.exp.name, str(num).zfill(3))
                    print(exp_path, "already exists: move to", exp_dir_name)
                else:
                    break
                exp_path = base_root / exp_dir / exp_dir_name
                
            self.experiment_dir = str(exp_path)
            os.makedirs(self.experiment_dir)
            print(f"Experiment directory: {self.experiment_dir}")

            with open(os.path.join(self.experiment_dir, "config.yaml"), "w") as f:
                omegaconf.OmegaConf.save(config=self.config, f=f.name)

            with open(os.path.join(self.experiment_dir, "run_command.sh"), "w") as f:
                f.write(" ".join(sys.argv))
                f.write("\n")

            self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
            os.mkdir(self.metrics_dir)
            self.inference_results_dir = os.path.join(
                self.experiment_dir, "inference_results"
            )
            os.mkdir(self.inference_results_dir)
        
        # 在分布式环境中同步实验目录路径
        if self.config.dist.enabled:
            # 创建一个临时tensor存储experiment_dir的长度
            if is_main_process:
                dir_path = self.experiment_dir
                dir_length = torch.tensor(len(dir_path), dtype=torch.int64, device=self.device)
            else:
                dir_length = torch.tensor(0, dtype=torch.int64, device=self.device)
                
            # 广播目录长度
            torch.distributed.broadcast(dir_length, src=0)
            
            # 非主进程接收目录路径
            if not is_main_process:
                dir_path = ""
            
            # 将目录路径转换为张量进行广播
            if is_main_process:
                dir_tensor = torch.tensor([ord(c) for c in dir_path], dtype=torch.int64, device=self.device)
            else:
                dir_tensor = torch.zeros(dir_length.item(), dtype=torch.int64, device=self.device)
            
            # 广播目录路径
            torch.distributed.broadcast(dir_tensor, src=0)
            
            # 非主进程解码目录路径
            if not is_main_process:
                dir_path = ''.join([chr(i) for i in dir_tensor.cpu().numpy()])
                self.experiment_dir = dir_path
                self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
                self.inference_results_dir = os.path.join(
                    self.experiment_dir, "inference_results"
                )
            
            # 确保所有进程同步
            torch.distributed.barrier()

    def _setup_metrics(self):
        metrics_names = self.config.train.val_metrics

        self.metrics = []
        for metric_name in metrics_names:
            metric_args = {}
            if hasattr(self.config.metrics, metric_name):
                metric_args = getattr(self.config.metrics, metric_name)
            self.metrics.append(metrics_registry[metric_name](**metric_args))


    def to_train(self):
        self.method.train()

    def to_eval(self):
        self.method.eval()

    def run(self):
        iter_info = StreamingMeans()
        self.to_train()

        for self.global_step in range(self.start_step, self.config.train.steps + 1):
            # 在每个epoch设置采样器的epoch
            if self.config.dist.enabled and self.train_sampler is not None:
                self.train_sampler.set_epoch(self.global_step)
            
            with Timer(iter_info, "iter_train"):
                loss_dict = self.train_step()
                
                # 如果是分布式训练，同步所有进程的损失
                if self.config.dist.enabled:
                    # 获取进程数量作为平均因子
                    world_size = self.config.dist.world_size
                    # 遍历损失字典的每个值，进行同步
                    for k, v in loss_dict.items():
                        if isinstance(v, float):
                            tensor = torch.tensor(v, device=self.device)
                            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                            loss_dict[k] = tensor.item() / world_size
                            
                iter_info.update({f"iter_train/{k}": v for k, v in loss_dict.items()})

            # 验证和日志记录
            if self.global_step % self.config.train.val_step == 0:
                with Timer(iter_info, "iter_val"):
                    # 所有进程需要参与验证，但只有主进程记录结果
                    val_loss_dict = self.validate()
                    
                    # 只在主进程更新验证指标
                    if not self.config.dist.enabled or self.config.dist.rank == 0:
                        iter_info.update({f"iter_val/{k}": v for k, v in val_loss_dict.items()})
                    
                    # 执行特殊验证并获取结果
                    orig_pics, method_pics, captions = self.inference_special()
                    
                    # TrainingLogger内部会处理是否为主进程
                    self.logger.save_validation_logs(
                        orig_pics,
                        method_pics, 
                        captions, 
                        special_paths=self.special_paths
                    )

            # 记录和保存检查点 - TrainingLogger和save_checkpoint内部会处理是否为主进程
            if self.global_step % self.config.train.log_step == 0:
                self.logger.save_train_logs(iter_info, self.global_step)
                # 确保在分布式环境中同步所有进程
                if self.config.dist.enabled:
                    torch.distributed.barrier()
                iter_info.clear()

            if self.global_step % self.config.train.checkpoint_step == 0:
                self.save_checkpoint()
                # 确保在分布式环境中同步所有进程
                if self.config.dist.enabled:
                    torch.distributed.barrier()

    def train_step(self):
        # 在每次step开始时打印当前的步骤信息
        if self.global_step % 2 == 0 and (not self.config.dist.enabled or self.config.dist.rank == 0):  # 每2步打印一次，避免输出过多，并且只在主进程打印
            progress = self.global_step / self.config.train.steps * 100
            print(f"Step {self.global_step}/{self.config.train.steps} ({progress:.2f}%)")
        
        x = next(self.train_dataloader)
        x = x.to(self.device).float()
        output = self.forward(x)

        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])

        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)

        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step and (not self.config.dist.enabled or self.config.dist.rank == 0):
                print("Start training with discriminator")
            if self.train_dataloader.batch_size != self.config.model.batch_size:
                if not self.config.dist.enabled or self.config.dist.rank == 0:
                    print(f"Changing batch size from {self.train_dataloader.batch_size} to {self.config.model.batch_size}")
                self.setup_dataloaders(self.config.model.batch_size)

            # 获取判别器并训练，处理DDP情况
            discriminator = self.get_model_component('discriminator')
            toogle_grad(discriminator, True)
            discriminator.train()

            disc_loss, disc_losses_dict = self.loss_builder.disc_loss(
                discriminator, 
                output["to_disc"]
                )
            loss_dict.update(disc_losses_dict)

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            toogle_grad(discriminator, False)
            discriminator.eval()

        # 确保latent_avg不参与反向传播
        base_model = self.get_base_model()
        base_model.latent_avg = base_model.latent_avg.detach()

        return loss_dict

    def save_checkpoint(self):
        # 只有主进程保存检查点
        if self.config.dist.enabled and self.config.dist.rank != 0:
            return
            
        save_name = f"iteration_{self.global_step}.pt"
        checkpoint_path = os.path.join(self.experiment_dir, save_name)
        save_dict = self.get_save_dict()
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(save_dict, checkpoint_path)

        options_path = os.path.join(self.experiment_dir, "save_options.json")
        save_options = {"start_step": self.global_step + 1}

        if self.config.exp.wandb:
            save_options.update(self.logger.wandb_logger.wandb_args)

        with open(options_path, "w") as f:
            json.dump(save_options, f)

    def get_save_dict(self):
        # 获取原始模型以正确保存latent_avg
        base_model = self.get_base_model()
        
        save_dict = {
            "state_dict": self.method.state_dict(),
            "encoder_opt": self.encoder_optimizer.state_dict(),
            "latent_avg": base_model.latent_avg
        }

        if self.config.train.train_dis:
            save_dict["disc_opt"] = self.disc_optimizer.state_dict()
        return save_dict

    @torch.inference_mode()
    def inference_special(self):
        # 只在主进程打印信息
        if not self.config.dist.enabled or self.config.dist.rank == 0:
            print("Running inversion for special")
        self.validate(special=True)
        
        # 在分布式模式下，只有主进程处理度量计算和结果收集
        if self.config.dist.enabled and self.config.dist.rank != 0:
            # 非主进程返回空结果
            return [], [], {}

        captions = defaultdict(str)
        for metric in self.metrics:
            if metric.get_name() == "FID":
                continue

            from_data_arg = {
                "fake_data": self.val_pics_res,
                "inp_data": self.val_pics_orig,
                "paths": self.special_paths,
            }
            metric_data, _, _ = metric(
                None, None, out_path=None, from_data=from_data_arg
            )
            
            # 确保metric_data有所有路径的键
            for path in self.special_paths:
                basename = os.path.basename(path)
                if basename in metric_data:
                    metric_value = metric_data[basename]
                    captions[path] += f"{metric.get_name()}: {metric_value:.3}\n"
                else:
                    # 如果找不到这个路径，添加一个占位符
                    captions[path] += f"{metric.get_name()}: N/A\n"

        return self.val_pics_orig, self.val_pics_res, captions

    @torch.inference_mode()
    def validate(self, special=False):
        if not special and (not self.config.dist.enabled or self.config.dist.rank == 0):
            print("Start validating")

        self.to_eval()
        
        # 在主进程中初始化结果列表
        if not self.config.dist.enabled or self.config.dist.rank == 0:
            self.val_pics_res = []
            self.val_pics_orig = []
        else:
            # 非主进程不需要收集图像，只参与计算
            self.val_pics_res = []  # 使用空列表而不是None，避免属性不存在的错误
            self.val_pics_orig = []

        if not special:
            dataloader = self.test_dataloader
            paths = self.paths
        else:
            dataloader = self.special_dataloader
            paths = self.special_paths

        # 记录本进程处理的样本索引，用于调试
        batch_indices = []
        
        global_i = 0
        for input_batch in tqdm(dataloader, disable=self.config.dist.enabled and self.config.dist.rank != 0):
            input_batch = input_batch.to(self.device).float()
            result_batch = self._run_on_batch(input_batch)
            
            # 记录当前批次的索引
            batch_size = input_batch.shape[0]
            indices = list(range(global_i, global_i + batch_size))
            batch_indices.extend(indices)
            global_i += batch_size
                
            # 只在主进程收集结果
            if not self.config.dist.enabled or self.config.dist.rank == 0:
                for i in range(result_batch.shape[0]):
                    idx = global_i - batch_size + i  # 计算当前样本的全局索引
                    
                    result = tensor2im(result_batch[i])
                    img = Image.fromarray(np.array(result)).convert("RGB")

                    memory_tmp = BytesIO()
                    img.save(memory_tmp, format="jpeg")
                    img = Image.open(memory_tmp).convert("RGB")
                    memory_tmp.close()

                    self.val_pics_res.append(img)
                    
                    # 确保索引在路径范围内
                    if idx < len(paths):
                        self.val_pics_orig.append(
                            Image.open(paths[idx]).convert("RGB")
                        )
                    else:
                        print(f"Warning: index {idx} out of range for paths (length {len(paths)})")

        # 在分布式设置中，我们需要确保所有进程完成验证
        if self.config.dist.enabled:
            if self.config.dist.rank == 0:
                print(f"Rank 0 processed {len(batch_indices)} samples with indices: {batch_indices[:10]}...")
            torch.distributed.barrier()

        metrics_dict = {}
        if not special and (not self.config.dist.enabled or self.config.dist.rank == 0):
            # 确保收集到的图像数量正确
            if len(self.val_pics_res) != len(self.val_pics_orig):
                print(f"Warning: Mismatch between val_pics_res ({len(self.val_pics_res)}) and val_pics_orig ({len(self.val_pics_orig)})")
                # 取二者中较小的数量
                min_count = min(len(self.val_pics_res), len(self.val_pics_orig))
                self.val_pics_res = self.val_pics_res[:min_count]
                self.val_pics_orig = self.val_pics_orig[:min_count]
                
            for metric in self.metrics:
                from_data_arg = {
                    "fake_data": self.val_pics_res,
                    "inp_data": self.val_pics_orig,
                    "paths": paths[:len(self.val_pics_res)],  # 确保路径数量匹配
                }
                _, metric_mean, _ = metric(
                    None, None, out_path=None, from_data=from_data_arg
                )
                metrics_dict[metric.get_name()] = metric_mean

        self.to_train()
        return metrics_dict

    @abstractmethod
    def _run_on_batch(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    def _setup_device(self):
        # 原有代码
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _setup_method(self):
        # 先初始化模型
        method_name = self.config.model.method
        self.method = methods_registry[method_name](
            checkpoint_path=self.config.model.checkpoint_path,
            **self.config.methods_args[method_name],
        ).to(self.device)
        
        # 如果启用分布式训练，进行DDP包装
        if self.config.dist.enabled:
            # 转换为SyncBatchNorm（如果配置中启用）
            if self.config.dist.sync_bn:
                self.method = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.method)
            
            # 包装模型为DDP
            self.method = torch.nn.parallel.DistributedDataParallel(
                self.method,
                device_ids=[self.config.dist.rank],
                output_device=self.config.dist.rank,
                find_unused_parameters=self.config.dist.find_unused_parameters
            )


@training_runners.add_to_registry(name="fse_inverter")
class FSEInverterTrainingRunner(BaseTrainingRunner):
    def forward(self, x):
        # 获取原始模型，处理DDP包装情况
        base_model = self.get_base_model()
        
        y_hat_inv, w_inv, fused_feat, w_feat = self.method(
            x,
            return_latents=True,
            n_iter=self.global_step
        )
                                                
        y_hat_inv_w, _ = base_model.decoder(
            [w_inv],
            input_is_latent=True,
            is_stylespace=False,
            randomize_noise=False
        )

        y_hat = torch.cat([y_hat_inv, y_hat_inv_w], dim=0)
                                   
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        if use_adv_loss:
            output["encoder"]["fake_preds"] = base_model.discriminator(y_hat, None)
            output["to_disc"]["y_hat"] = y_hat
            output["to_disc"]["x"] = x
            output["to_disc"]["step"] = self.global_step
        
        y_hat = base_model.pool(y_hat)
        x = base_model.pool(x)
        x = torch.cat([x, x], dim=0)
        
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat
        output["encoder"]["feat_recon"] = fused_feat
        output["encoder"]["feat_real"] = w_feat

        return output

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch


@training_runners.add_to_registry(name="fse_editor")
class FSEEditorTrainingRunner(BaseTrainingRunner):
    def forward(self, x):
        # 获取原始模型，处理DDP包装情况
        base_model = self.get_base_model()
        
        # get inversion batch
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)
        # w变量未被使用，可以立即删除
        del w, fused_feat, w_feat

        # get editing batch
        with torch.no_grad():
            # sample X_E as training input and X'_E as training target
            d, strenght = get_random_edit()
            
            x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

            # 使用e4e编码器生成潜在编码
            w_e4e = base_model.e4e_encoder(x_resh)
            w_e4e = w_e4e + base_model.latent_avg
            x_E, fx_e4e = base_model.decoder(
                [w_e4e],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False,
                return_features=True
            )

            # 第二阶段-第一部分-编辑阶段
            # get_edited_latent尤其要进行更换
            edited_w_e4e = self.get_edited_latent(w_e4e, d, [strenght])
            # w_e4e已不再使用
            del w_e4e
            
            if isinstance(edited_w_e4e, tuple):
                # stylespace case
                y_E, fy_e4e = base_model.decoder(
                    edited_w_e4e, 
                    is_stylespace=True, 
                    input_is_latent=True,
                    randomize_noise=False,
                    return_features=True
                )
            else:
                edited_w_e4e = torch.cat(edited_w_e4e, dim=0)
                y_E, fy_e4e = base_model.decoder(
                    [edited_w_e4e], 
                    is_stylespace=False,
                    input_is_latent=True,
                    randomize_noise=False,
                    return_features=True
                )
            # edited_w_e4e已不再使用
            del edited_w_e4e

            y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False) # X'_E
            x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False) # X_E
            # x_E已不再使用
            del x_E
            
            delta = fx_e4e[9] - fy_e4e[9]
            # fx_e4e和fy_e4e已不再使用
            del fx_e4e, fy_e4e
            torch.cuda.empty_cache()  # 释放一部分显存

            if d in self.config.train.disc_edits:
                x_E_256 = torch.cat([x_E_256, x_resh], dim=0)
                delta = torch.cat([delta, delta], dim=0)
            
            # 第二阶段-第二部分
            # 使用特征提取backbone
            w_x_E, x_E_predicted_feats = base_model.inverter.fs_backbone(x_E_256)
            w_x_E = w_x_E + base_model.latent_avg
            
            w_x_E_edited = self.get_edited_latent(w_x_E, d, [strenght])
            is_stylespace = isinstance(w_x_E_edited, tuple)
            if not is_stylespace:
                w_x_E_edited = [torch.cat(w_x_E_edited, dim=0)]
            
            # 使用解码器获取特征
            _, x_E_w_feats = base_model.decoder(
                [w_x_E],
                input_is_latent=True,
                return_features=True,
                is_stylespace=False,
                randomize_noise=False,
                early_stop=64
            )
            # w_x_E已不再使用
            del w_x_E
            
            x_E_w_feat = x_E_w_feats[9] 
            # x_E_w_feats除了索引9外已不再使用
            del x_E_w_feats
            
            to_fuser = torch.cat([x_E_predicted_feats, x_E_w_feat], dim=1)
            # x_E_predicted_feats和x_E_w_feat已不再使用
            del x_E_predicted_feats, x_E_w_feat
            
            # 融合特征
            x_E_fused_feat = base_model.inverter.fuser(to_fuser)
            # to_fuser已不再使用
            del to_fuser
        
        # delta的使用在这里
        to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
        # x_E_fused_feat和delta已不再使用
        del x_E_fused_feat, delta
        
        # 使用编码器生成编辑特征
        x_E_edited_feat = base_model.encoder(to_feature_editor)
        # to_feature_editor已不再使用
        del to_feature_editor
        
        x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)
        # x_E_edited_feat已包含在列表中，可以删除原引用
        del x_E_edited_feat
        
        # 在最耗内存的decoder调用前释放一些内存
        torch.cuda.empty_cache()

        # 生成最终的编辑图像
        y_hat_edit, _ = base_model.decoder(
            w_x_E_edited,
            input_is_latent=True,
            new_features=x_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=is_stylespace,
            randomize_noise=False
        )
        
        # w_x_E_edited和x_E_edited_feats已不再使用
        del w_x_E_edited, x_E_edited_feats
        
        # 损失计算
        bs = x_resh.size(0)
        output = {"encoder": {}, "to_disc": {}}
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        if use_adv_loss:
            if x_E_256.size(0) > x_resh.size(0):
                assert y_hat_edit.size(0) == bs * 2
                output["encoder"]["fake_preds"] = base_model.discriminator(
                    torch.cat([y_hat_inv, y_hat_edit[bs:]], dim=0), 
                    None
                )
            else:
                output["encoder"]["fake_preds"] = base_model.discriminator(y_hat_inv, None)
            output["to_disc"]["y_hat"] = y_hat_inv
            output["to_disc"]["x"] = x
            output["to_disc"]["step"] = self.global_step
        
        if x_E_256.size(0) > x_resh.size(0):
            assert y_hat_edit.size(0) == bs * 2
            y_hat_edit = y_hat_edit[:bs]

        x = torch.cat([x, y_E], dim=0)
        # y_E已不再使用
        del y_E
        
        y_hat = torch.cat([y_hat_inv, y_hat_edit])
        # y_hat_inv和y_hat_edit已不再使用
        del y_hat_inv, y_hat_edit
        
        # 下采样输出图像
        y_hat = base_model.pool(y_hat)
        x = base_model.pool(x)
        output["encoder"]["x"] = x
        output["encoder"]["y_hat"] = y_hat

        # 最后再清理一次内存
        torch.cuda.empty_cache()
        
        return output

    def _run_on_batch(self, inputs):
        result_batch = self.method(inputs)
        return result_batch
