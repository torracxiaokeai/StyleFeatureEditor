import os
import json
import wandb
import time

import torch
import numpy as np
from abc import abstractmethod 
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm.auto import tqdm
from io import BytesIO
import torch.nn.functional as F

from PIL import Image
from pathlib import Path

from utils.class_registry import ClassRegistry
from datasets.datasets import ImageDataset
from datasets.transforms import transforms_registry
from utils.common_utils import tensor2im
from runners.base_runner import BaseRunner
from training.loggers import BaseTimer
from utils.common_utils import get_keys
from metrics.metrics import metrics_registry


inference_runner_registry = ClassRegistry()


@inference_runner_registry.add_to_registry(name="base_inference_runner")
class BaseInferenceRunner(BaseRunner):
    def run(self):
        self.run_inversion()
        self.run_editing()

    @torch.inference_mode()
    def run_inversion(self):
        output_inv_dir =  Path(self.config.exp.output_dir) / "inversion"
        output_inv_dir.mkdir(parents=True, exist_ok=True)


        transform_dict = transforms_registry[self.config.data.transform]().get_transforms()
        dataset = ImageDataset(self.config.data.inference_dir, transform_dict["test"])
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.model.workers,
        )

        self.method_results = []
        self.paths = dataset.paths
        self.method.eval()

        print("Start inversion")
        global_i = 0

        for input_batch in tqdm(dataloader):
            input_cuda = input_batch.to(self.device).float()

            images, result_batch = self._run_on_batch(input_cuda)
            result_batch["img_names"] = []
            
            for tensor in images:
                image = tensor2im(tensor)
                img_name = os.path.basename(dataset.paths[global_i])
                result_batch["img_names"].append(img_name)
                image.save(output_inv_dir / img_name)
                global_i += 1

            self.method_results.append(result_batch)


    @torch.inference_mode()
    def run_editing(self):
        """
        处理源图像和驱动图像进行编辑。
        简化版：处理单张源图像与多个驱动图像。
        """
        # 检查配置中是否指定了驱动图像目录
        if not hasattr(self.config.inference, "driver_dir") or not self.config.inference.driver_dir:
            print("未指定驱动图像目录，跳过编辑阶段")
            return
        
        # 加载驱动图像
        driver_dir = self.config.inference.driver_dir
        output_edit_dir = Path(self.config.exp.output_dir) / "edited"
        output_edit_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用相同的变换加载驱动图像
        transform_dict = transforms_registry[self.config.data.transform]().get_transforms()
        driver_dataset = ImageDataset(driver_dir, transform_dict["test"])
        driver_dataloader = DataLoader(
            driver_dataset,
            batch_size=1,  # 一次处理一个驱动图像
            shuffle=False,
            num_workers=self.config.model.workers,
        )
        
        print(f"开始编辑，使用驱动图像目录: {driver_dir}")
        
        # 获取源图像信息（仅使用第一个源图像）
        if not self.method_results or len(self.method_results) == 0:
            print("错误：未找到源图像结果")
            return
            
        source_result = self.method_results[0]  # 使用第一个批次
        source_name = source_result["img_names"][0] if "img_names" in source_result else "source"
        
        # 创建源图像目录
        source_output_dir = output_edit_dir / f"source_{source_name.split('.')[0]}"
        source_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 对每个驱动图像进行处理
        for driver_idx, driver_batch in enumerate(tqdm(driver_dataloader, desc="处理驱动图像")):
            driver_cuda = driver_batch.to(self.device).float()
            driver_name = os.path.basename(driver_dataset.paths[driver_idx])
            
            # 保存驱动图像
            driver_img = tensor2im(driver_cuda[0])
            driver_img.save(source_output_dir / f"driver_{driver_name}")
            
            # 应用驱动图像到源图像
            edited_imgs = self._run_editing_on_batch(source_result, driver_cuda)
            
            # 保存编辑后的图像
            if edited_imgs and len(edited_imgs) > 0:
                edited_img_tensor = edited_imgs[0][0]  # 取第一个结果
                edited_img_pil = tensor2im(edited_img_tensor)
                save_path = source_output_dir / f"edited_by_{driver_name}"
                edited_img_pil.save(save_path)
                print(f"保存编辑图像到 {save_path}")
            else:
                print(f"警告：驱动图像 {driver_name} 编辑失败")

    @abstractmethod
    def _run_on_batch(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def _run_editing_on_batch(self, method_res_batch, driver_batch):
        raise NotImplementedError()


@inference_runner_registry.add_to_registry(name="fse_inference_runner")
class FSEInferenceRunner(BaseInferenceRunner):
    def setup(self):
        super().setup()
        # 加载ViVFace模型
        self.vivface_model = self._load_vivface_model()
    
    def _load_vivface_model(self):
        # 加载预训练的ViVFace模型
        from models.vivface.psp_identity_related import pSp
        import torch
        import argparse
        
        # 加载ViVFace预训练检查点
        checkpoint_path = self.config.inference.vivface_checkpoint
        print(f'从ViVFace检查点加载模型: {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 创建模型配置
        opts = ckpt['opts']
        opts['checkpoint_path'] = checkpoint_path
        opts['device'] = self.device
        opts = argparse.Namespace(**opts)
        
        # 创建pSp模型
        vivface_model = pSp(opts)
        vivface_model.eval()  # 设为评估模式
        vivface_model = vivface_model.to(self.device)
        
        # 冻结参数
        for param in vivface_model.parameters():
            param.requires_grad = False
            
        return vivface_model
    
    def _run_on_batch(self, inputs):
        # 获取原始图像的反演结果
        images, w_recon, fused_feat, predicted_feat = self.method(inputs, return_latents=True)
        
        # 使用ViVFace提取特征
        x_resh = F.interpolate(inputs, size=(256, 256), mode="bilinear", align_corners=False)
        with torch.no_grad():
            # 提取w向量(身份)和ss向量(表情)
            latent = self.vivface_model.forward(x_resh, return_latents=True, return_images=False)
            w_vivface = latent['w_latent']
            ss_vivface = latent['ss_latent']
        
        result_batch = {
            "latents": w_recon, 
            "fused_feat": fused_feat, 
            "predicted_feat": predicted_feat,
            "w_vivface": w_vivface,
            "ss_vivface": ss_vivface,
            "inputs": inputs.cpu()
        }
        
        return images, result_batch
          
    def _run_editing_on_batch(self, method_res_batch, driver_batch, mask=None):
        """
        执行编辑操作 - 简化版，专注于处理单张源图像和单张驱动图像
        
        Args:
            method_res_batch: 包含源图像信息的字典
            driver_batch: 驱动图像批次 (1, C, H, W)
            mask: 可选的遮罩
            
        Returns:
            dict: 包含编辑后图像和中间结果的字典
                - edited_images: 最终编辑结果
                - source_recon: 使用源图像身份和表情重建的图像
                - identity_driven: 使用源图像身份和驱动图像表情生成的图像
        """
        base_model = self.method  # 获取模型
        edited_images = []
        n_iter = 1e5
        
        # 获取源图像信息（假设只处理批次中的第一个图像）
        i = 0  # 只处理索引为0的图像
        
        # 源图像信息
        source_w_vivface = method_res_batch["w_vivface"][i].unsqueeze(0)  # 源图像的身份特征
        source_ss_vivface = method_res_batch["ss_vivface"][i].unsqueeze(0)  # 源图像的表情特征
        source_fused_feat = method_res_batch["fused_feat"][i].to(self.device).unsqueeze(0)  # 源图像的融合特征
        
        # 驱动图像信息
        driver_resh = F.interpolate(driver_batch, size=(256, 256), mode="bilinear", align_corners=False)
        with torch.no_grad():
            driver_latent = self.vivface_model.forward(driver_resh, return_latents=True, return_images=False)
            driver_ss_vivface = driver_latent['ss_latent']  # 驱动图像的表情特征
        
        # 步骤1: 使用源图像的身份和表情特征生成源图像
        x_E, fx_e4e = base_model.decoder(
            [source_w_vivface],
            input_is_latent=True,
            randomize_noise=False,
            return_features=True,
            ss_latent=source_ss_vivface  # 使用源图像的表情特征
        )
        
        # 步骤2: 使用源图像的身份和驱动图像的表情特征生成编辑图像
        y_E, fy_e4e = base_model.decoder(
            [source_w_vivface],
            input_is_latent=True,
            randomize_noise=False,
            return_features=True,
            ss_latent=driver_ss_vivface  # 使用驱动图像的表情特征
        )
        
        # 步骤3: 计算特征差异
        delta = fx_e4e[9] - fy_e4e[9]
        
        # 步骤4: 对源图像使用特征提取backbone
        x_E_256 = F.interpolate(x_E, size=(256, 256), mode="bilinear", align_corners=False)
        w_x_E, x_E_predicted_feats = base_model.inverter.fs_backbone(x_E_256)
        w_x_E = w_x_E + base_model.latent_avg
        
        # 步骤5: 使用解码器获取特征
        _, x_E_w_feats = base_model.decoder(
            [w_x_E],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )
        
        x_E_w_feat = x_E_w_feats[9]
        
        # 步骤6: 融合特征
        to_fuser = torch.cat([x_E_predicted_feats, x_E_w_feat], dim=1)
        x_E_fused_feat = base_model.inverter.fuser(to_fuser)
        
        # 步骤7: 使用Feature Editor编辑特征
        to_feature_editor = torch.cat([x_E_fused_feat, delta], dim=1)
        x_E_edited_feat = base_model.encoder(to_feature_editor)
        
        # 步骤8: 准备编辑特征
        x_E_edited_feats = [None] * 9 + [x_E_edited_feat] + [None] * (17 - 9)
        
        # 步骤9: 生成最终的编辑图像
        w_x_E_edited = [w_x_E]  # 使用源图像的w_x_E
        is_stylespace = False
        
        # 如果有遮罩，应用遮罩
        if mask is not None:
            image_edits, _ = base_model.decoder(
                w_x_E_edited,
                input_is_latent=True,
                new_features=x_E_edited_feats,
                feature_scale=min(1.0, 0.0001 * n_iter),
                is_stylespace=is_stylespace,
                randomize_noise=False,
                ss_latent=driver_ss_vivface  # 使用驱动图像的表情特征
            )
            
            # 调整遮罩尺寸以匹配图像
            if mask.shape[2:] != image_edits.shape[2:]:
                mask = F.interpolate(mask, size=image_edits.shape[2:], mode='bilinear', align_corners=False)
            
            # 获取原始图像用于混合
            orig_image = method_res_batch["inputs"][i].to(self.device).unsqueeze(0)
            
            # 将原图调整为与编辑图像相同的尺寸
            if orig_image.shape[2:] != image_edits.shape[2:]:
                orig_image = F.interpolate(orig_image, size=image_edits.shape[2:], mode='bilinear', align_corners=False)
            
            # 应用遮罩混合
            image_edits = mask * image_edits + (1 - mask) * orig_image
        else:
            image_edits, _ = base_model.decoder(
                w_x_E_edited,
                input_is_latent=True,
                new_features=x_E_edited_feats,
                feature_scale=min(1.0, 0.0001 * n_iter),
                is_stylespace=is_stylespace,
                randomize_noise=False,
                ss_latent=driver_ss_vivface  # 使用驱动图像的表情特征
            )
        
        edited_images.append(image_edits)
        
        # 返回包含最终编辑结果和中间结果的字典
        return {
            "edited_images": edited_images,  # 最终编辑结果
            "source_recon": x_E,  # 源图像重建 (使用源身份+源表情)
            "identity_driven": y_E,  # 身份驱动图像 (使用源身份+驱动表情)
        }


@inference_runner_registry.add_to_registry(name="fse_inverter_inference_runner")
class FSEInverterInferenceRunner(BaseInferenceRunner):
    def _run_on_batch(self, inputs):
        images, w_recon, fused_feat, predicted_feat = self.method(inputs, return_latents=True)
        result_batch = {
            "latents": w_recon, 
            "fused_feat": fused_feat, 
            "predicted_feat": predicted_feat,
            "inputs": inputs.cpu()
        }
        
        return images, result_batch
          
    def _run_editing_on_batch(self, method_res_batch, editing_name, editing_degrees):
        orig_latents = method_res_batch["latents"]
        edited_images = []
        n_iter = 1e5

        for i, latent in enumerate(orig_latents):
            edited_latents = self.get_edited_latent(
                latent.unsqueeze(0), 
                editing_name, 
                editing_degrees, 
                method_res_batch["inputs"][i].unsqueeze(0)
            )

            if edited_latents is None:
                print(f"WARNING, skip editing {editing_name}")
                continue

            is_stylespace = isinstance(edited_latents, tuple)
            if not is_stylespace:
                edited_latents = torch.cat(edited_latents, dim=0).unsqueeze(0)

            w_latent = latent.unsqueeze(0).repeat(len(editing_degrees), 1, 1)

            fused_feat = method_res_batch["fused_feat"][i].to(self.device)
            fused_feat = fused_feat.repeat(len(editing_degrees), 1, 1, 1)

            edit_features = [None] * 9 + [fused_feat] + [None] * (17 - 9)

            image_edits, _ = self.method.decoder(
                edited_latents,
                input_is_latent=True,
                new_features=edit_features,
                feature_scale=min(1.0, 0.0001 * n_iter),
                is_stylespace=is_stylespace,
                randomize_noise=False
            )

            edited_images.append(image_edits)
        edited_images = torch.stack(edited_images)

        return edited_images  # : torch.tensor(batch_size x len(powers) x pics)


