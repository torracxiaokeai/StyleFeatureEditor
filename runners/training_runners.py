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
        # 如果已经初始化过并且有self.method属性，则直接返回
        if hasattr(self, '_initialized') and self._initialized and hasattr(self, 'method'):
            print("Runner已完全初始化，跳过setup流程")
            return
            
        # 添加初始化标记属性
        if not hasattr(self, '_initialized'):
            self._initialized = False
            
        self.start_step = self.config.train.start_step
        self._setup_device()
        
        # 只在未初始化时设置实验目录
        if not self._initialized:
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

        print("start_batch_size: " + str(start_batch_size))
        print("self.config.train.bs_used_before_adv_loss: " + str(self.config.train.bs_used_before_adv_loss))
        print("self.config.train.train_dis: " + str(self.config.train.train_dis))
        print("self.config.model.batch_size: " + str(self.config.model.batch_size))

        self._setup_dataloaders(start_batch_size)

        self._setup_latent_editor()
        self._setup_optimizers()
        self._setup_loss()
        
        # 设置初始化完成标记
        self._initialized = True

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
        # 如果已经初始化过，则跳过
        if hasattr(self, '_initialized') and self._initialized:
            print("数据集已初始化，跳过重复加载")
            return
            
        print("Loading dataset")
        transform_dict = transforms_registry[self.config.data.transform]().get_transforms()
        
        # 根据配置选择数据集类型
        dataset_type = getattr(self.config.data, "dataset_type", "image")
        
        if dataset_type == "vivface":
            from datasets.datasets import ViVFaceDataset
            print("Using ViVFaceDataset for training and validation")
            
            self.train_dataset = ViVFaceDataset(
                self.config.data.input_train_dir, transform_dict["train"]
            )
            
            self.test_dataset = ViVFaceDataset(
                self.config.data.input_val_dir, transform_dict["test"]
            )
            
            self.special_dataset = ViVFaceDataset(
                self.config.data.special_dir, transform_dict["test"]
            )
            
            # 为了兼容性，我们仍然需要设置paths
            self.paths = []
            self.special_paths = []
        elif dataset_type == "vivface_edit":
            from datasets.datasets import ViVFaceEditDataset
            print("Using ViVFaceEditDataset for training and validation")
            
            self.train_dataset = ViVFaceEditDataset(
                self.config.data.input_train_dir,  # 使用input_train_dir作为唯一数据源
                None,  # 不再需要driver_dir
                transform_dict["train"]
            )
            
            # 对于测试/验证，我们仍然使用普通的ImageDataset
            self.test_dataset = ImageDataset(
                self.config.data.input_val_dir, transform_dict["test"]
            )
            
            self.special_dataset = ImageDataset(
                self.config.data.special_dir, transform_dict["test"]
            )
            
            self.paths = self.test_dataset.paths if hasattr(self.test_dataset, 'paths') else []
            self.special_paths = self.special_dataset.paths if hasattr(self.special_dataset, 'paths') else []
        else:
            # 默认使用普通的ImageDataset
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
        # 重写损失设置以添加额外的ViVFace特定损失
        enc_losses_dict = self.config.encoder_losses
        disc_losses_dict = self.config.disc_losses
        
        # 这里可以添加特定于ViVFace的损失
        # 例如身份一致性损失、表情一致性损失等

        self.loss_builder = LossBuilder(
            enc_losses_dict, 
            disc_losses_dict, 
            self.device
        )

    def _setup_experiment_dir(self):
        # 如果已经初始化过，则跳过
        if hasattr(self, '_initialized') and self._initialized:
            print("实验目录已初始化，跳过重复创建")
            return
            
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
        """
        重写train_step方法，适应ViVFaceDataset返回的字典格式数据，并添加渐进式训练检查
        """
        # 在每次step开始时打印当前的步骤信息
        if self.global_step % 2 == 0 and (not self.config.dist.enabled or self.config.dist.rank == 0):  # 每2步打印一次，避免输出过多，并且只在主进程打印
            progress = self.global_step / self.config.train.steps * 100
            print(f"Step {self.global_step}/{self.config.train.steps} ({progress:.2f}%)")
        
        # 获取批次数据
        batch = next(self.train_dataloader)

        # print("batch: " + str(batch))    
        
        # 如果是字典类型，需要将每个张量移到正确的设备
        if isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device).float()
        else:
            # 如果不是字典，按原来的方式处理
            batch = batch.to(self.device).float()
        
        # 前向传播
        output = self.forward(batch)
        
        # =========================================
        # ❶ 计算 encoder 相关损失（判别器不参与梯度）
        # =========================================
        # 先拿到判别器实例——建议放在 forward 之后立即执行
        discriminator = self.get_model_component('discriminator')

        toogle_grad(discriminator, False)         # 彻底关闭判别器梯度
        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])
        
        # 反向传播
        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)
        
        # =========================================
        # ❷ 计算并更新判别器（重新打开梯度）
        # =========================================
        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step and \
            (not self.config.dist.enabled or self.config.dist.rank == 0):
                print("Start training with discriminator")

            # dataloader 批次大小检查（保持你原来的逻辑）
            if self.train_dataloader.batch_size * self.config.dist.world_size  != self.config.model.batch_size:
                if not self.config.dist.enabled or self.config.dist.rank == 0:
                    print(f"Changing batch size from {self.train_dataloader.batch_size}"
                        f" to {self.config.model.batch_size}")
                self._setup_dataloaders(self.config.model.batch_size)

            # ——真正训练判别器——
            toogle_grad(discriminator, True)      # 打开梯度
            discriminator.train()

            disc_loss, disc_losses_dict = self.loss_builder.disc_loss(
                discriminator,
                output["to_disc"],
            )
            loss_dict.update(disc_losses_dict)

            self.disc_optimizer.zero_grad()
            disc_loss.backward()                  # ← 这里才第一次标记判别器参数
            self.disc_optimizer.step()

            toogle_grad(discriminator, False)     # 立即关闭，后续其他模块用不到
            discriminator.eval()
        
        # 确保latent_avg不参与反向传播
        base_model = self.get_base_model()
        base_model.latent_avg = base_model.latent_avg.detach()
        
        # 检查是否需要更新渐进式训练阶段
        if self.enable_progressive_training:
            self.check_for_progressive_training_update()
        
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

        # 确保数据一致性：图片数量和路径数量应该匹配
        if not self.config.dist.enabled or self.config.dist.rank == 0:
            # 检查并修正数据不匹配的问题
            val_pics_count = len(self.val_pics_orig)
            processed_paths_count = len(self.processed_paths)
            special_paths_count = len(self.special_paths)
            
            print(f"Validation pics: {val_pics_count}, Processed paths: {processed_paths_count}, Special paths: {special_paths_count}")
            
            # 如果处理的图片数量少于special_paths，使用processed_paths
            if val_pics_count < special_paths_count:
                print(f"Using processed_paths ({processed_paths_count}) instead of special_paths ({special_paths_count})")
                actual_paths = self.processed_paths[:val_pics_count]  # 确保路径数量不超过图片数量
            else:
                actual_paths = self.special_paths[:val_pics_count]  # 确保路径数量不超过图片数量

        captions = defaultdict(str)
        for metric in self.metrics:
            if metric.get_name() == "FID":
                continue

            # 使用实际处理过的样本的路径
            sample_paths = actual_paths

            from_data_arg = {
                "fake_data": self.val_pics_res[:len(sample_paths)],  # 确保数据长度匹配
                "inp_data": self.val_pics_orig[:len(sample_paths)],  # 确保数据长度匹配
                "paths": sample_paths,
            }
            
            try:
                metric_data, _, _ = metric(
                    None, None, out_path=None, from_data=from_data_arg
                )
                
                # 确保metric_data有所有路径的键
                for path in sample_paths:
                    basename = os.path.basename(path)
                    if basename in metric_data:
                        metric_value = metric_data[basename]
                        captions[path] += f"{metric.get_name()}: {metric_value:.3}\n"
                    else:
                        # 如果找不到这个路径，添加一个占位符
                        captions[path] += f"{metric.get_name()}: N/A\n"
            except Exception as e:
                print(f"Error calculating {metric.get_name()}: {e}")
                # 为每个样本添加错误信息
                for path in sample_paths:
                    captions[path] += f"{metric.get_name()}: Error\n"

        # 返回匹配的数据
        result_pics_orig = self.val_pics_orig[:len(actual_paths)]
        result_pics_res = self.val_pics_res[:len(actual_paths)]
        
        return result_pics_orig, result_pics_res, captions

    @torch.inference_mode()
    def validate(self, special=False):
        if not special and (not self.config.dist.enabled or self.config.dist.rank == 0):
            print("Start validating")

        self.to_eval()
        
        # 在主进程中初始化结果列表
        if not self.config.dist.enabled or self.config.dist.rank == 0:
            self.val_pics_res = []
            self.val_pics_orig = []
            # 保存处理过的样本的路径
            self.processed_paths = []
        else:
            # 非主进程不需要收集图像，只参与计算
            self.val_pics_res = []  # 使用空列表而不是None，避免属性不存在的错误
            self.val_pics_orig = []
            self.processed_paths = []

        if not special:
            dataloader = self.test_dataloader
            paths = self.paths
        else:
            dataloader = self.special_dataloader
            paths = self.special_paths

        # 记录本进程处理的样本索引，用于调试
        batch_indices = []
        
        # 限制验证样本数量，避免处理过多数据
        # 验证时只使用前100个样本
        max_val_samples = 100
        sample_count = 0
        
        global_i = 0
        for input_batch in tqdm(dataloader, disable=self.config.dist.enabled and self.config.dist.rank != 0):
            # 检查是否已达到最大样本数
            if sample_count >= max_val_samples:
                break
                
            # 检查输入是否为字典格式（ViVFaceDataset返回的格式）
            if isinstance(input_batch, dict):
                original_image = input_batch['source']
                if 'source_path' in input_batch:
                    current_paths = [input_batch['source_path']]
                else:
                    current_paths = []
            else:
                original_image = input_batch
                current_paths = []
            
            # 确保原始图像保存在CPU上用于后续处理
            original_image_cpu = original_image
            if isinstance(original_image_cpu, torch.Tensor) and original_image_cpu.device != torch.device('cpu'):
                original_image_cpu = original_image_cpu.to('cpu')
            
            # 运行模型 - _run_on_batch内部会确保数据移动到正确的设备
            result_batch = self._run_on_batch(input_batch)
            
            # 确保结果移回CPU用于后续处理
            if isinstance(result_batch, torch.Tensor) and result_batch.device != torch.device('cpu'):
                result_batch = result_batch.to('cpu')
            
            # 记录当前批次的索引
            batch_size = result_batch.shape[0]
            indices = list(range(global_i, global_i + batch_size))
            batch_indices.extend(indices)
            global_i += batch_size
            sample_count += batch_size
                
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
                    
                    # 添加原始图像
                    if isinstance(input_batch, dict) and 'source' in input_batch:
                        # 如果是字典格式，使用source图像
                        orig_img = tensor2im(original_image_cpu[i] if isinstance(original_image_cpu, torch.Tensor) else original_image_cpu)
                        self.val_pics_orig.append(Image.fromarray(np.array(orig_img)).convert("RGB"))
                        # 添加路径到processed_paths（如果有）
                        if 'source_path' in input_batch and i < len(input_batch['source_path']):
                            self.processed_paths.append(input_batch['source_path'][i])
                        else:
                            self.processed_paths.append(f"img_{idx}")
                    elif idx < len(paths):
                        # 否则使用paths中的路径
                        self.val_pics_orig.append(
                            Image.open(paths[idx]).convert("RGB")
                        )
                        # 添加路径到processed_paths
                        self.processed_paths.append(paths[idx])
                    else:
                        print(f"Warning: index {idx} out of range for paths (length {len(paths)})")
                        self.val_pics_orig.append(Image.new('RGB', (256, 256), color = 'black'))
                        self.processed_paths.append(f"img_{idx}")

        # 在需要使用barrier的地方，临时退出推理模式
        if self.config.dist.enabled:
            # 退出推理模式执行barrier
            with torch.inference_mode(False):
                if self.config.dist.rank == 0:
                    print(f"Rank 0 processed {len(batch_indices)} samples with indices: {batch_indices[:10]}...")
                torch.distributed.barrier()
        else:
            print(f"Processed {len(batch_indices)} samples for validation (limited from total {len(dataloader.dataset)})")

        metrics_dict = {}
        with torch.no_grad():
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
                        "paths": paths[:len(self.val_pics_res)] if len(paths) > 0 else ["img_" + str(i) for i in range(len(self.val_pics_res))],  # 确保路径数量匹配
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
        # 如果已经初始化过，则跳过
        if hasattr(self, '_initialized') and self._initialized:
            print("模型已初始化，跳过重复加载")
            return
            
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
            
            # 完全禁用DDP的缓冲区广播
            self.method = torch.nn.parallel.DistributedDataParallel(
                self.method,
                device_ids=[self.config.dist.rank],
                output_device=self.config.dist.rank,
                find_unused_parameters=self.config.dist.find_unused_parameters,
                broadcast_buffers=False  # 禁用缓冲区广播
            )

    def _setup_latent_editor(self):
        # 基础实现为空，子类可以根据需要覆盖此方法
        pass

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        """
        检查是否需要更新渐进式训练阶段
        
        Args:
            is_resume_from_ckpt: 是否从检查点恢复训练的检查
        """
        # FSEEditorTrainingRunner不使用渐进式训练
        # 这个方法只是为了兼容BaseTrainingRunner中的train_step方法
        pass


@training_runners.add_to_registry(name="fse_inverter")
class FSEInverterTrainingRunner(BaseTrainingRunner):
    def __init__(self, *args, **kwargs):
        super(FSEInverterTrainingRunner, self).__init__(*args, **kwargs)
        
        # 初始化渐进式训练设置 - FSE Inverter通常不使用渐进式训练
        self.enable_progressive_training = False
        if hasattr(self.config.train, "enable_progressive_training"):
            self.enable_progressive_training = self.config.train.enable_progressive_training
    
    def forward(self, inputs):
        # 处理输入参数 - 兼容字典和张量输入
        if isinstance(inputs, dict):
            x = inputs.get('source', inputs.get('driver', list(inputs.values())[0]))
        else:
            x = inputs
            
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
        # 确保输入是字典格式
        if not isinstance(inputs, dict):
            # 兼容旧格式，将单一张量转换为字典格式
            inputs = {'source': inputs, 'driver': inputs}
        
        # 确保数据在正确的设备上
        if 'source' in inputs:
            source = inputs['source']
            if isinstance(source, torch.Tensor) and source.device != self.device:
                inputs['source'] = source.to(self.device)
        
        if 'driver' in inputs:
            driver = inputs['driver']
            if isinstance(driver, torch.Tensor) and driver.device != self.device:
                inputs['driver'] = driver.to(self.device)
        
        # 检查是否处于验证模式
        if not self.method.training:
            # 验证模式：直接调用method进行推理，返回重建图像
            # 处理输入参数 - 兼容字典和张量输入
            if isinstance(inputs, dict):
                x = inputs.get('source', inputs.get('driver', list(inputs.values())[0]))
            else:
                x = inputs
            
            # 直接调用method进行推理
            result_batch = self.method(x, return_latents=False)
            return result_batch
        else:
            # 训练模式：调用forward方法处理输入，返回完整字典
            result_batch = self.forward(inputs)
            return result_batch


@training_runners.add_to_registry(name="fse_editor")
class FSEEditorTrainingRunner(BaseTrainingRunner):
    def __init__(self, *args, **kwargs):
        super(FSEEditorTrainingRunner, self).__init__(*args, **kwargs)
        
        # 首先初始化基础属性（device等）
        if not hasattr(self, '_initialized'):
            self._initialized = False
            
        self.start_step = self.config.train.start_step
        self._setup_device()
        
        # 在有了device之后，立即加载ViVFace模型，确保在_setup_optimizers执行时模型已存在
        if not hasattr(self, 'vivface_model'):
            self.vivface_model = self._load_vivface_model()
            
        # 然后调用父类setup，这会创建包括vivface_optimizer在内的所有优化器
        super().setup()
            
        # 初始化渐进式训练设置
        self.enable_progressive_training = False
        if hasattr(self.config.train, "enable_progressive_training"):
            self.enable_progressive_training = self.config.train.enable_progressive_training
            
        # 初始化渐进式训练步骤
        self.progressive_steps = []
        if hasattr(self.config.train, "progressive_steps"):
            self.progressive_steps = self.config.train.progressive_steps
        
    def _load_vivface_model(self):
        # 加载预训练的ViVFace模型
        from models.vivface.psp_identity_related import pSp
        import torch
        import argparse
        
        # 加载ViVFace预训练检查点
        checkpoint_path = self.config.train.vivface_checkpoint
        print(f'从ViVFace检查点加载模型: {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建模型配置
        opts = ckpt['opts']
        opts['checkpoint_path'] = checkpoint_path
        opts['device'] = self.device
        opts = argparse.Namespace(**opts)
        
        # 创建pSp模型
        vivface_model = pSp(opts)
        vivface_model.train()  # 设为训练模式，允许参与训练
        vivface_model = vivface_model.to(self.device)
        
        # 不再冻结参数，允许vivface_model参与训练
        # for param in vivface_model.parameters():
        #     param.requires_grad = False
        
        # 如果有保存的训练权重，加载它们
        if self.config.model.checkpoint_path != "":
            try:
                training_ckpt = torch.load(self.config.model.checkpoint_path, map_location='cpu')
                if "vivface_model" in training_ckpt:
                    print('加载已训练的ViVFace模型权重')
                    vivface_model.load_state_dict(training_ckpt["vivface_model"])
                else:
                    print('未找到已训练的ViVFace权重，使用预训练权重')
            except Exception as e:
                print(f'加载ViVFace训练权重失败: {e}，使用预训练权重')
            
        return vivface_model
        
    def _setup_optimizers(self):
        # 先调用父类方法设置基础优化器
        super()._setup_optimizers()
        
        # 为vivface_model添加优化器
        if hasattr(self, 'vivface_model') and self.vivface_model is not None:
            # 获取vivface_model的参数
            vivface_params = list(self.vivface_model.parameters())
            
            # 使用与encoder相同的优化器配置
            optimizer_args = dict(
                self.config.optimizers[self.config.train.encoder_optimizer]
            )
            optimizer_args["params"] = vivface_params
            
            self.vivface_optimizer = optimizers[self.config.train.encoder_optimizer](
                **optimizer_args
            )
            
            # 如果有checkpoint，加载vivface优化器状态
            if self.config.model.checkpoint_path != "":
                try:
                    ckpt = torch.load(self.config.model.checkpoint_path, map_location="cpu")
                    if "vivface_opt" in ckpt.keys():
                        self.vivface_optimizer.load_state_dict(ckpt["vivface_opt"])
                        print('成功加载ViVFace优化器状态')
                    else:
                        print('WARNING: 未找到ViVFace优化器状态，使用默认初始化')
                except Exception as e:
                    print(f'WARNING: 加载ViVFace优化器状态失败: {e}')
        else:
            print('WARNING: vivface_model未初始化，跳过优化器设置')
            
    def get_save_dict(self):
        # 获取基础保存字典
        save_dict = super().get_save_dict()
        
        # 添加vivface_model的权重
        if hasattr(self, 'vivface_model') and self.vivface_model is not None:
            save_dict["vivface_model"] = self.vivface_model.state_dict()
            print('保存ViVFace模型权重到检查点')
        
        # 添加vivface优化器状态
        if hasattr(self, 'vivface_optimizer') and self.vivface_optimizer is not None:
            save_dict["vivface_opt"] = self.vivface_optimizer.state_dict()
            print('保存ViVFace优化器状态到检查点')
            
        return save_dict
    
    def forward(self, inputs):
        # 假设inputs是一个字典，包含source和driver
        if not isinstance(inputs, dict) or 'source' not in inputs or 'driver' not in inputs:
            raise ValueError("输入必须是包含'source'和'driver'键的字典")
            
        # 获取源图像和驱动图像
        x = inputs['source']  # 源图像S
        driver = inputs['driver']  # 驱动图像D2

        # 获取原始模型，处理DDP包装情况
        base_model = self.get_base_model()

        # 第一步：获取原始图像的反演结果
        y_hat_inv, w, fused_feat, w_feat = self.method(x, return_latents=True)
        del w, fused_feat, w_feat  # 释放不需要的变量
        
        # 第二步：使用ViVFace提取身份和表情特征
        # 对源图像和驱动图像进行大小调整
        x_resh = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        driver_resh = F.interpolate(driver, size=(256, 256), mode="bilinear", align_corners=False)
        
        # 从源图像提取w向量（身份特征）和ss向量（表情特征）
        S_latent = self.vivface_model.forward(x_resh, return_latents=True, return_images=False)
        w_S = S_latent['w_latent']
        ss_S = S_latent['ss_latent']
        
        # 从驱动图像提取ss向量（表情特征）
        D2_latent = self.vivface_model.forward(driver_resh, return_latents=True, return_images=False)
        ss_D2 = D2_latent['ss_latent']
        
        # 使用身份和表情特征生成编辑后的图像
        x_E, fx_e4e = base_model.decoder(
            [w_S],
            input_is_latent=True,
            randomize_noise=False,
            return_features=True,
            ss_latent=ss_S  # 使用源图像的ss特征
        )
        del ss_S
        
        # 使用w_S和ss_D2生成编辑后的图像
        y_E, fy_e4e = base_model.decoder(
            [w_S],
            input_is_latent=True,
            randomize_noise=False,
            return_features=True,
            ss_latent=ss_D2  # 使用驱动图像的表情特征
        )
        
        # 计算特征差异
        y_E_256 = F.interpolate(y_E, size=(256, 256), mode="bilinear", align_corners=False)
        x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False)
        del x_E, y_E_256
        
        delta = fx_e4e[9] - fy_e4e[9]
        del fx_e4e, fy_e4e
        torch.cuda.empty_cache()
        
        # 处理鉴别器编辑（保持原有逻辑的兼容）
        if hasattr(self.config.train, 'disc_edits') and len(self.config.train.disc_edits) > 0:
            x_E_256 = torch.cat([x_E_256, x_resh], dim=0)
            delta = torch.cat([delta, delta], dim=0)
        
        # 使用特征提取backbone
        w_x_E, x_E_predicted_feats = base_model.inverter.fs_backbone(x_E_256)
        w_x_E = w_x_E + base_model.latent_avg
        
        # 使用ss_D2编辑w_x_E
        # 注意这里不再使用随机编辑方向，而是使用ss_D2作为编辑方向
        w_x_E_edited = [w_x_E]  # 使用原始w作为基础
        # 将StyleSpace格式标记为True
        is_stylespace = True
        
        # 使用解码器获取特征
        _, x_E_w_feats = base_model.decoder(
            [w_x_E],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )
        del w_x_E
        
        x_E_w_feat = x_E_w_feats[9]
        del x_E_w_feats
        
        to_fuser = torch.cat([x_E_predicted_feats, x_E_w_feat], dim=1)
        del x_E_predicted_feats, x_E_w_feat
        
        # 融合特征
        x_E_fused_feat = base_model.inverter.fuser(to_fuser)
        del to_fuser
        
        # 使用delta特征进行编辑
        to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
        del x_E_fused_feat, delta
        
        # 使用编码器生成编辑特征
        x_E_edited_feat = base_model.encoder(to_feature_editor)
        del to_feature_editor
        
        x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)
        del x_E_edited_feat
        
        torch.cuda.empty_cache()
        
        # 使用ViVFace模式而不是StyleSpace模式
        is_stylespace = False
        
        # 生成最终的编辑图像，使用ss_D2作为StyleSpace输入
        y_hat_edit, _ = base_model.decoder(
            w_x_E_edited,
            input_is_latent=True,
            new_features=x_E_edited_feats,
            feature_scale=1.0,
            is_stylespace=is_stylespace,
            randomize_noise=False,
            ss_latent=ss_D2  # 使用驱动图像的表情特征
        )
        
        del w_x_E_edited, x_E_edited_feats
        
        # 以下部分保持不变，进行损失计算
        # (后续代码保持原样)
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

        # 确保x和y_E具有相同的空间尺寸
        # x经过了pool处理（在最后），所以我们需要确保y_E也经过相同处理
        # 将y_E保存到一个临时变量，以便我们可以在连接后删除原始变量
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
        # 检查输入是否为字典格式（ViVFaceDataset返回的格式）
        if isinstance(inputs, dict):
            # 对于验证，我们只使用source图像
            if 'source' in inputs:
                # 确保数据在正确的设备上
                source = inputs['source']
                if isinstance(source, torch.Tensor) and source.device != self.device:
                    source = source.to(self.device)
                result_batch = self.method(source)  # 移除randomize_noise参数
                return result_batch
        
        # 处理传统格式的输入（普通图像张量）
        # 确保数据在正确的设备上
        if isinstance(inputs, torch.Tensor) and inputs.device != self.device:
            inputs = inputs.to(self.device)
        result_batch = self.method(inputs)  # 移除randomize_noise参数
        return result_batch

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        """
        检查是否需要更新渐进式训练阶段
        
        Args:
            is_resume_from_ckpt: 是否从检查点恢复训练的检查
        """
        # FSEEditorTrainingRunner不使用渐进式训练
        # 这个方法只是为了兼容BaseTrainingRunner中的train_step方法
        pass


@training_runners.add_to_registry(name="vivface_training")
class ViVFaceTrainingRunner(BaseTrainingRunner):
    """
    用于训练ViVFace模型的第一阶段（W+训练）
    
    该阶段主要训练E4E编码器以生成高质量的w_latent和ss_latent，
    实现身份与表情的解耦。
    """
    def __init__(self, config):
        # 调用父类初始化
        super(ViVFaceTrainingRunner, self).__init__(config)
        
        # 确保模型和基础组件已初始化
        if not hasattr(self, 'method'):
            # 如果method尚未初始化，需要显式调用setup
            self.setup()
        
        # 初始化额外的ViVFace特定属性
        # 初始化global_step
        self.global_step = self.config.train.start_step
        
        # 设置编码器的训练阶段
        if hasattr(self.config.train, "progressive_stage"):
            stage_name = self.config.train.progressive_stage
            base_model = self.get_base_model()
            encoder = base_model.encoder
            
            # 将字符串转换为枚举值
            stage = getattr(ProgressiveStage, stage_name)
            encoder.set_progressive_stage(stage)
            
            print(f"ViVFace训练阶段设置为: {stage_name}")
        
        # 初始化是否需要进行渐进式训练更新的检查
        self.enable_progressive_training = False
        if hasattr(self.config.train, "enable_progressive_training"):
            self.enable_progressive_training = self.config.train.enable_progressive_training
        
        # 初始化渐进式训练步骤
        self.progressive_steps = []
        if hasattr(self.config.train, "progressive_steps"):
            self.progressive_steps = self.config.train.progressive_steps
            print(f"渐进式训练步骤设置为: {self.progressive_steps}")
            
        # 如果从检查点恢复训练，立即检查并更新训练阶段
        if self.enable_progressive_training and self.global_step > 0:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
    
    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        """
        检查是否需要更新渐进式训练阶段
        
        Args:
            is_resume_from_ckpt: 是否从检查点恢复训练的检查
        """
        if not self.enable_progressive_training:
            return
            
        if not self.progressive_steps:
            return
            
        # 获取基础模型和编码器
        base_model = self.get_base_model()
        encoder = base_model.encoder
        
        # 检查每个进度阶段
        for i, step in enumerate(self.progressive_steps):
            # 从检查点恢复时，如果当前步骤已经超过了特定的进度阶段，直接设置为该阶段
            if is_resume_from_ckpt and self.global_step >= step:
                if i < len(ProgressiveStage):  # 确保索引不超出ProgressiveStage范围
                    encoder.set_progressive_stage(ProgressiveStage(i))
                    print(f"从检查点恢复：更新渐进式训练阶段至 {ProgressiveStage(i)}")
            
            # 在正常训练中，当达到特定步骤时更新训练阶段
            if self.global_step == step:
                if i < len(ProgressiveStage):  # 确保索引不超出ProgressiveStage范围
                    encoder.set_progressive_stage(ProgressiveStage(i))
                    print(f"更新渐进式训练阶段至 {ProgressiveStage(i)}")
    
    def forward(self, batch):
        # 获取原始模型，处理DDP包装情况
        base_model = self.get_base_model()
        
        # 从批次中提取三种图像
        S = batch['source']       # 源图像
        D1 = batch['same_id']     # 相同身份不同表情
        D2 = batch['diff_id']     # 不同身份
        
        # 1. 自重建路径 (S→S_hat)
        w_S, ss_S = base_model.encoder(S)
        
        # 确保latent_avg在与w_S相同的设备上
        latent_avg = base_model.latent_avg.to(w_S.device)
        w_S = w_S + latent_avg.unsqueeze(0).repeat(w_S.shape[0], 1, 1)
        
        S_hat, _ = base_model.decoder([w_S], input_is_latent=True, ss_latent=ss_S)
        # 将S_hat从1024x1024降采样到256x256
        S_hat_downsampled = F.interpolate(S_hat, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 释放原始分辨率的S_hat
        del S_hat
        torch.cuda.empty_cache()
        
        # 2. 同身份表情迁移 (S+D1→S_D1)
        _, ss_D1 = base_model.encoder(D1)
        S_D1, _ = base_model.decoder([w_S], input_is_latent=True, ss_latent=ss_D1)
        # 将S_D1从1024x1024降采样到256x256
        S_D1_downsampled = F.interpolate(S_D1, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 删除不再需要的变量
        del S_D1, ss_D1
        torch.cuda.empty_cache()
        
        # 3. 中性表情生成 (S→S_neutral)
        ss_zero = torch.zeros_like(ss_S)
        S_neutral, _ = base_model.decoder([w_S], input_is_latent=True, ss_latent=ss_zero)
        # 将S_neutral从1024x1024降采样到256x256
        S_neutral_downsampled = F.interpolate(S_neutral, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 删除不再需要的变量
        del S_neutral, ss_zero
        torch.cuda.empty_cache()
        
        # 4. 跨身份身份迁移 (D2+S→D2_S)
        _, ss_D2 = base_model.encoder(D2)
        
        
        # 使用S的身份(w_S)和D2的表情(ss_D2)创建D2_S
        D2_S, _ = base_model.decoder([w_S], input_is_latent=True, ss_latent=ss_D2)
        # 将D2_S从1024x1024降采样到256x256
        D2_S_downsampled = F.interpolate(D2_S, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 删除原始分辨率的D2_S
        del D2_S
        torch.cuda.empty_cache()
        
        # 获取D2_S的编码用于一致性损失
        w_D2_S, ss_D2_S = base_model.encoder(D2_S_downsampled)
        
        # 确保latent_avg在与w_D2_S相同的设备上
        latent_avg = base_model.latent_avg.to(w_D2_S.device)
        w_D2_S = w_D2_S + latent_avg.unsqueeze(0).repeat(w_D2_S.shape[0], 1, 1)

        # 构建适用于LossBuilder的输出格式
        output = {"encoder": {}, "to_disc": {}}
        
        # 基本图像和编码数据
        output["encoder"]["source"] = S
        output["encoder"]["same_id"] = D1
        output["encoder"]["y_hat_s"] = S_hat_downsampled
        output["encoder"]["y_hat_s_d1"] = S_D1_downsampled
        output["encoder"]["y_hat_s_neutral"] = S_neutral_downsampled
        output["encoder"]["y_hat_d2_s"] = D2_S_downsampled
        output["encoder"]["w_s"] = w_S
        output["encoder"]["ss_s"] = ss_S
        output["encoder"]["ss_d2"] = ss_D2
        
        # 新增：D2_S的潜在编码，用于一致性损失
        output["encoder"]["w_d2_s"] = w_D2_S
        output["encoder"]["ss_d2_s"] = ss_D2_S
        
        # 获取当前进度阶段信息（用于delta损失）
        output["encoder"]["progressive_stage"] = base_model.encoder.progressive_stage
        
        # 对抗损失相关设置
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        
        if use_adv_loss:
            # 添加判别器相关数据
            discriminator = base_model.discriminator
            
            # 避免一次性创建太大的张量，分步处理
            # 注意：这里我们只使用降采样后的图像
            S_hat_D1 = torch.cat([S_hat_downsampled, S_D1_downsampled], dim=0)
            S_neutral_D2_S = torch.cat([S_neutral_downsampled, D2_S_downsampled], dim=0)
            
            # 最后再合并
            concat_images = torch.cat([S_hat_D1, S_neutral_D2_S], dim=0)
            
            # 删除中间变量以节省内存
            del S_hat_D1, S_neutral_D2_S
            
            # 在执行判别器操作前再次清除缓存
            torch.cuda.empty_cache()

            c = torch.zeros(concat_images.size(0), 0, device=concat_images.device)
            
            # 直接使用256x256分辨率图像进行判别 - 修改后的判别器已支持256x256输入
            output["encoder"]["fake_preds"] = discriminator(concat_images, c)
            output["to_disc"]["y_hat"] = concat_images
            output["to_disc"]["x"] = torch.cat([S, D1, S, D2], dim=0)
            output["to_disc"]["c"] = c
            output["to_disc"]["step"] = self.global_step
        
        # 最后一次清理缓存，确保返回前释放不必要的内存
        torch.cuda.empty_cache()
        
        return output

    def _run_on_batch(self, inputs):
        # 检查输入是否为字典格式（ViVFaceDataset返回的格式）
        if isinstance(inputs, dict):
            # 对于验证，我们只使用source图像
            if 'source' in inputs:
                # 确保数据在正确的设备上
                source = inputs['source']
                if isinstance(source, torch.Tensor) and source.device != self.device:
                    source = source.to(self.device)
                result_batch = self.method(source)  # 移除randomize_noise参数
                return result_batch
        
        # 处理传统格式的输入（普通图像张量）
        # 确保数据在正确的设备上
        if isinstance(inputs, torch.Tensor) and inputs.device != self.device:
            inputs = inputs.to(self.device)
        result_batch = self.method(inputs)  # 移除randomize_noise参数
        return result_batch
    
    def _setup_loss(self):
        # 重写损失设置以添加额外的ViVFace特定损失
        enc_losses_dict = self.config.encoder_losses
        disc_losses_dict = self.config.disc_losses
        
        # 这里可以添加特定于ViVFace的损失
        # 例如身份一致性损失、表情一致性损失等

        self.loss_builder = LossBuilder(
            enc_losses_dict, 
            disc_losses_dict, 
            self.device
        )

    def train_step(self):
        """
        重写train_step方法，适应ViVFaceDataset返回的字典格式数据，并添加渐进式训练检查
        """
        # 在每次step开始时打印当前的步骤信息
        if self.global_step % 2 == 0 and (not self.config.dist.enabled or self.config.dist.rank == 0):  # 每2步打印一次，避免输出过多，并且只在主进程打印
            progress = self.global_step / self.config.train.steps * 100
            print(f"VivFace Step {self.global_step}/{self.config.train.steps} ({progress:.2f}%)")
        
        # 获取批次数据
        batch = next(self.train_dataloader)

        # 如果是字典类型，需要将每个张量移到正确的设备
        if isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device).float()
        else:
            # 如果不是字典，按原来的方式处理
            batch = batch.to(self.device).float()
        
        # 前向传播
        output = self.forward(batch)
        
        # =========================================
        # ❶ 计算 encoder 相关损失（判别器不参与梯度）
        # =========================================
        # 先拿到判别器实例——建议放在 forward 之后立即执行
        if hasattr(self.get_base_model(), 'discriminator'):
            discriminator = self.get_model_component('discriminator')
            toogle_grad(discriminator, False)         # 彻底关闭判别器梯度
        
        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])
        
        # 反向传播
        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)
        
        # =========================================
        # ❷ 训练W判别器（如果启用）
        # =========================================
        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step and \
            (not self.config.dist.enabled or self.config.dist.rank == 0):
                print("Start training with W discriminator")

            # dataloader 批次大小检查（保持你原来的逻辑）
            if self.train_dataloader.batch_size * self.config.dist.world_size  != self.config.model.batch_size:
                if not self.config.dist.enabled or self.config.dist.rank == 0:
                    print(f"Changing batch size from {self.train_dataloader.batch_size}"
                        f" to {self.config.model.batch_size}")
                self._setup_dataloaders(self.config.model.batch_size)

            # ——真正训练W判别器——
            if hasattr(self.get_base_model(), 'discriminator'):
                discriminator = self.get_model_component('discriminator')
                toogle_grad(discriminator, True)      # 打开梯度
                discriminator.train()

                disc_loss, disc_losses_dict = self.loss_builder.disc_loss(
                    discriminator,
                    output["to_disc"],
                )
                loss_dict.update(disc_losses_dict)

                self.disc_optimizer.zero_grad()
                disc_loss.backward()                  # ← 这里才第一次标记判别器参数
                self.disc_optimizer.step()

                toogle_grad(discriminator, False)     # 立即关闭，后续其他模块用不到
                discriminator.eval()
        
        # 确保latent_avg不参与反向传播
        base_model = self.get_base_model()
        if hasattr(base_model, 'latent_avg') and base_model.latent_avg is not None:
            base_model.latent_avg = base_model.latent_avg.detach()
        
        # 检查是否需要更新渐进式训练阶段
        if self.enable_progressive_training:
            self.check_for_progressive_training_update()
        
        # 清除缓存
        torch.cuda.empty_cache()
        
        return loss_dict


@training_runners.add_to_registry(name="vivface_method_training")
class VivFaceMethodTrainingRunner(BaseTrainingRunner):
    """
    基于真实ViVFace代码实现的完整训练Runner
    
    这是参考ViVFace原始训练代码coach_first_stage_complete_dp.py，
    在StyleFeatureEditor框架中重新实现的版本。
    
    主要特点：
    1. 完整的四种重建路径：自重建、同身份表情迁移、中性表情生成、跨身份身份迁移
    2. 多组件损失函数：重建损失、一致性损失、正则化损失、身份保持损失、渐进式delta损失
    3. 渐进式训练支持
    4. W判别器对抗训练
    """
    def __init__(self, config):
        # 调用父类初始化
        super(VivFaceMethodTrainingRunner, self).__init__(config)
        
        # 确保模型和基础组件已初始化
        if not hasattr(self, 'method'):
            # 如果method尚未初始化，需要显式调用setup
            self.setup()
        
        # 初始化额外的VivFace特定属性
        # 初始化global_step
        self.global_step = self.config.train.start_step
        
        # 设置编码器的训练阶段
        if hasattr(self.config.train, "progressive_stage"):
            stage_name = self.config.train.progressive_stage
            base_model = self.get_base_model()
            encoder = base_model.encoder
            
            # 将字符串转换为枚举值
            stage = getattr(ProgressiveStage, stage_name)
            encoder.set_progressive_stage(stage)
            
            print(f"VivFace训练阶段设置为: {stage_name}")
        
        # 初始化是否需要进行渐进式训练更新的检查
        self.enable_progressive_training = False
        if hasattr(self.config.train, "enable_progressive_training"):
            self.enable_progressive_training = self.config.train.enable_progressive_training
        
        # 初始化渐进式训练步骤
        self.progressive_steps = []
        if hasattr(self.config.train, "progressive_steps"):
            self.progressive_steps = self.config.train.progressive_steps
            print(f"渐进式训练步骤设置为: {self.progressive_steps}")
            
        # 如果从检查点恢复训练，立即检查并更新训练阶段
        if self.enable_progressive_training and self.global_step > 0:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
    
    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        """
        检查是否需要更新渐进式训练阶段
        
        Args:
            is_resume_from_ckpt: 是否从检查点恢复训练的检查
        """
        if not self.enable_progressive_training:
            return
            
        if not self.progressive_steps:
            return
            
        # 获取基础模型和编码器
        base_model = self.get_base_model()
        encoder = base_model.encoder
        
        # 检查每个进度阶段
        for i, step in enumerate(self.progressive_steps):
            # 从检查点恢复时，如果当前步骤已经超过了特定的进度阶段，直接设置为该阶段
            if is_resume_from_ckpt and self.global_step >= step:
                if i < len(ProgressiveStage):  # 确保索引不超出ProgressiveStage范围
                    encoder.set_progressive_stage(ProgressiveStage(i))
                    print(f"从检查点恢复：更新渐进式训练阶段至 {ProgressiveStage(i)}")
            
            # 在正常训练中，当达到特定步骤时更新训练阶段
            if self.global_step == step:
                if i < len(ProgressiveStage):  # 确保索引不超出ProgressiveStage范围
                    encoder.set_progressive_stage(ProgressiveStage(i))
                    print(f"更新渐进式训练阶段至 {ProgressiveStage(i)}")
    
    def forward(self, batch):
        """
        VivFace的前向传播函数
        
        基于原始ViVFace的coach_first_stage_complete_dp.py中的forward方法实现。
        实现四种重建路径的完整训练流程。
        
        Args:
            batch: 包含source、driving、other_identity的批次数据
            
        Returns:
            包含encoder和to_disc信息的输出字典
        """
        # 获取原始模型，处理DDP包装情况
        base_model = self.get_base_model()
        
        # 从批次中提取三种图像
        S = batch['source']       # 源图像
        D1 = batch['same_id']     # 相同身份不同表情 (D1在ViVFace中对应same_id)
        D2 = batch['diff_id']     # 不同身份 (D2在ViVFace中对应diff_id)
        
        print("hello1")
        # 1. 自重建路径 (S→S_hat)
        S_hat, S_latent = base_model.forward(S, return_latents=True)
        
        # 2. 同身份表情迁移 (S+D1→S_D1)
        # 先获取D1的编码
        D_hat, D1_latent = base_model.forward(D1, return_latents=True)
        print("hello2")
        # 使用S的身份编码和D1的表情编码生成S_D1
        S_D1 = base_model.forward(
            S, 
            skip_latent=True, 
            w_latent=S_latent['w_latent'],
            input_memory=S_latent['input_memory'],
            ss_generic_latent=D1_latent['ss_generic_latent'], 
            return_latents=False
        )
        print("hello3")
        # 3. 中性表情生成 (S→S_neutral) - 只在启用身份损失时计算
        zero_ss_latent = torch.zeros_like(S_latent['ss_latent']).to(S.device)
        if self.config.encoder_losses.get('id_lambda', 0.0) != 0.0:
            S_neutral = base_model.forward(
                S, 
                skip_latent=True,
                w_latent=S_latent['w_latent'],
                ss_generic_latent=zero_ss_latent,
                input_memory=S_latent['input_memory'],
                return_latents=False
            )
            
            # 4. 跨身份身份迁移 (D2+S→D2_S)
            # 使用S的身份编码和D2的上下文生成D2_S
            D2_S, D2_latent = base_model.forward(
                D2, 
                w_latent=S_latent['w_latent'], 
                return_latents=True, 
                input_memory=S_latent['input_memory']
            )
        else:
            S_neutral = None
            D2_S = None
            D2_latent = None
        
        print("hello4")
        # 5. 获取D2_S的重编码（用于一致性损失）
        if D2_S is not None:
            D2_S_latent = base_model.forward(D2_S, return_latents=True, return_images=False)
        else:
            D2_S_latent = None
        
        # 构建适用于LossBuilder的输出格式
        output = {"encoder": {}, "to_disc": {}}
        
        # 基本图像和编码数据
        output["encoder"]["S"] = S  # 源图像
        output["encoder"]["D1"] = D1  # 相同身份不同表情
        output["encoder"]["D2"] = D2  # 不同身份
        output["encoder"]["S_hat"] = S_hat  # 自重建
        output["encoder"]["S_D1"] = S_D1  # 同身份表情迁移
        output["encoder"]["S_neutral"] = S_neutral  # 中性表情
        output["encoder"]["D2_S"] = D2_S  # 跨身份身份迁移
        
        # 潜在编码数据
        output["encoder"]["S_latent"] = S_latent
        output["encoder"]["D1_latent"] = D1_latent
        output["encoder"]["D2_latent"] = D2_latent
        output["encoder"]["D2_S_latent"] = D2_S_latent
        
        # 获取当前进度阶段信息（用于delta损失）
        output["encoder"]["progressive_stage"] = base_model.encoder.progressive_stage
        
        # 对抗损失相关设置
        use_adv_loss = (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        )
        output["encoder"]["use_adv_loss"] = use_adv_loss
        
        if use_adv_loss:
            # 添加判别器相关数据
            discriminator = base_model.discriminator
            
            # 准备用于判别器的图像
            # 根据原始ViVFace实现，使用四种生成的图像
            if S_neutral is not None and D2_S is not None:
                concat_images = torch.cat([S_hat, S_D1, S_neutral, D2_S], dim=0)
                real_images = torch.cat([S, D1, S, D2], dim=0)
            else:
                # 如果没有身份损失，只使用前两种重建
                concat_images = torch.cat([S_hat, S_D1], dim=0)
                real_images = torch.cat([S, D1], dim=0)
            
            # 创建条件张量（如果判别器需要）
            c = torch.zeros(concat_images.size(0), 0, device=concat_images.device)
            
            # 使用W判别器判别w_latent
            output["encoder"]["fake_preds"] = S_latent['w_latent']  # 传递w_latent给LossBuilder处理
            output["to_disc"]["y_hat"] = concat_images
            output["to_disc"]["x"] = real_images
            output["to_disc"]["c"] = c
            output["to_disc"]["step"] = self.global_step
        
        return output

    def _run_on_batch(self, inputs):
        """
        在批次上运行模型（用于验证）
        
        Args:
            inputs: 输入数据，可以是字典或张量
            
        Returns:
            模型输出
        """
        # 检查输入是否为字典格式（ViVFaceDataset返回的格式）
        if isinstance(inputs, dict):
            # 对于验证，我们只使用source图像进行自重建
            if 'source' in inputs:
                # 确保数据在正确的设备上
                source = inputs['source']
                if isinstance(source, torch.Tensor) and source.device != self.device:
                    source = source.to(self.device)
                result_batch = self.method(source, return_latents=False)
                return result_batch
        
        # 处理传统格式的输入（普通图像张量）
        # 确保数据在正确的设备上
        if isinstance(inputs, torch.Tensor) and inputs.device != self.device:
            inputs = inputs.to(self.device)
        result_batch = self.method(inputs, return_latents=False)
        return result_batch
    
    def _setup_loss(self):
        """
        设置VivFace特定的损失函数
        
        重写基类方法，添加VivFace特定的损失配置
        """
        enc_losses_dict = self.config.encoder_losses
        disc_losses_dict = self.config.disc_losses
        
        # 这里可以添加特定于VivFace的损失验证和配置
        # 确保必要的损失函数被配置
        required_losses = ['L_self', 'L_reenact', 'L_latent_consistency', 'L_ss_latent_consistency']
        for loss_name in required_losses:
            if loss_name not in enc_losses_dict:
                print(f"警告：缺少必要的损失函数配置: {loss_name}")

        self.loss_builder = LossBuilder(
            enc_losses_dict, 
            disc_losses_dict, 
            self.device
        )

    def train_step(self):
        """
        重写train_step方法，适应VivFace的完整训练流程
        
        基于原始ViVFace的coach_first_stage_complete_dp.py中的训练步骤实现
        """
        # 在每次step开始时打印当前的步骤信息
        if self.global_step % 2 == 0 and (not self.config.dist.enabled or self.config.dist.rank == 0):
            progress = self.global_step / self.config.train.steps * 100
            print(f"VivFace Step {self.global_step}/{self.config.train.steps} ({progress:.2f}%)")
        
        # 获取批次数据
        batch = next(self.train_dataloader)
        
        # 如果是字典类型，需要将每个张量移到正确的设备
        if isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device).float()
        else:
            # 如果不是字典，按原来的方式处理
            batch = batch.to(self.device).float()
        
        # 前向传播
        output = self.forward(batch)
        
        # =========================================
        # ❶ 计算 encoder 相关损失（判别器不参与梯度）
        # =========================================
        # 先拿到判别器实例——建议放在 forward 之后立即执行
        if hasattr(self.get_base_model(), 'discriminator'):
            discriminator = self.get_model_component('discriminator')
            toogle_grad(discriminator, False)         # 彻底关闭判别器梯度
        
        enc_loss, loss_dict = self.loss_builder.encoder_loss(output["encoder"])
        
        # 反向传播
        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()
        loss_dict["enc_loss"] = float(enc_loss)
        
        # =========================================
        # ❷ 训练W判别器（如果启用）
        # =========================================
        if (
            self.config.train.train_dis
            and self.global_step >= self.config.train.dis_train_start_step
        ):
            if self.global_step == self.config.train.dis_train_start_step and \
            (not self.config.dist.enabled or self.config.dist.rank == 0):
                print("Start training with W discriminator")

            # dataloader 批次大小检查（保持你原来的逻辑）
            if self.train_dataloader.batch_size * self.config.dist.world_size  != self.config.model.batch_size:
                if not self.config.dist.enabled or self.config.dist.rank == 0:
                    print(f"Changing batch size from {self.train_dataloader.batch_size}"
                        f" to {self.config.model.batch_size}")
                self._setup_dataloaders(self.config.model.batch_size)

            # ——真正训练W判别器——
            if hasattr(self.get_base_model(), 'discriminator'):
                discriminator = self.get_model_component('discriminator')
                toogle_grad(discriminator, True)      # 打开梯度
                discriminator.train()

                disc_loss, disc_losses_dict = self.loss_builder.disc_loss(
                    discriminator,
                    output["to_disc"],
                )
                loss_dict.update(disc_losses_dict)

                self.disc_optimizer.zero_grad()
                disc_loss.backward()                  # ← 这里才第一次标记判别器参数
                self.disc_optimizer.step()

                toogle_grad(discriminator, False)     # 立即关闭，后续其他模块用不到
                discriminator.eval()
        
        # 确保latent_avg不参与反向传播
        base_model = self.get_base_model()
        if hasattr(base_model, 'latent_avg') and base_model.latent_avg is not None:
            base_model.latent_avg = base_model.latent_avg.detach()
        
        # 检查是否需要更新渐进式训练阶段
        if self.enable_progressive_training:
            self.check_for_progressive_training_update()
        
        # 清除缓存
        torch.cuda.empty_cache()
        
        return loss_dict
