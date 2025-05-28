import math
import sys
import pickle
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.psp.encoders import psp_encoders
from models.psp.stylegan2.model import Generator
from models.hyperinverter.stylegan2_ada import Discriminator 
from utils.class_registry import ClassRegistry
from utils.common_utils import get_keys
from utils.model_utils import toogle_grad
from configs.paths import DefaultPaths
from argparse import Namespace
from training.loggers import BaseTimer
from models.psp.encoders.psp_encoders import ProgressiveStage


sys.path.append("./utils")
methods_registry = ClassRegistry()


# 添加全局辅助函数，用于处理module前缀问题
def get_keys_with_prefix_handling(ckpt, name):
    """处理状态字典中可能带有module前缀的键名
    
    Args:
        ckpt: 检查点字典
        name: 键名前缀（如"encoder"、"inverter"等）
        
    Returns:
        处理后的状态字典，移除了多余的前缀
    """
    # 先尝试使用原来的get_keys函数获取状态字典
    state_dict = get_keys(ckpt, name)
    
    # 如果状态字典为空或加载失败，可能是因为键名带有module前缀
    if not state_dict and 'state_dict' in ckpt:
        result = {}
        prefix = f'module.{name}.'
        for key, value in ckpt['state_dict'].items():
            # 处理带有module前缀的键
            if key.startswith(prefix):
                new_key = key[len(prefix):]  # 移除"module.encoder."或"module.inverter."等前缀
                result[new_key] = value
        return result
    
    return state_dict


@methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path"))
class FSEFull(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None,
                 inverter_pth=None):
        super(FSEFull, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.inverter_pth = inverter_pth

        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None
        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


    def load_disc(self):
        # We used the hyperinverter discriminator since it has a cars checkpoint
        print("Loading default Discriminator from ", self.opts.stylegan_weights_pkl)
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)

    def load_disc_from_ckpt(self, ckpt):
        # 检查是否有module.discriminator前缀的键
        has_module_prefix = any(key.startswith("module.discriminator.") for key in ckpt["state_dict"].keys())
        
        # 检查是否有discriminator前缀的键
        unique_keys = set(key.split(".")[0] for key in ckpt["state_dict"].keys())
        has_disc_prefix = "discriminator" in unique_keys
        
        # 情况1: 有module.discriminator前缀
        if has_module_prefix:
            print("检测到module.discriminator前缀，使用前缀处理方式加载判别器")
            disc_state_dict = get_keys_with_prefix_handling(ckpt, "discriminator")
            try:
                self.discriminator.load_state_dict(disc_state_dict, strict=True)
                print("成功加载判别器权重")
            except Exception as e:
                print(f"加载判别器权重时出错: {e}")
                print("尝试非严格模式加载...")
                self.discriminator.load_state_dict(disc_state_dict, strict=False)
        
        # 情况2: 有discriminator前缀（原始方式）
        elif has_disc_prefix:
            print("检测到discriminator前缀，使用原始方式加载判别器")
            try:
                self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
                print("成功加载判别器权重")
            except Exception as e:
                print(f"加载判别器权重时出错: {e}")
                print("尝试非严格模式加载...")
                self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=False)
        
        # 情况3: 两种前缀都没有找到
        else:
            print("未找到判别器权重，保留默认权重")

    def load_weights_for_inference(self, editor_ckpt_path, inverter_ckpt_path):
        """
        专门用于推理时的权重加载方法
        分别从editor checkpoint和inverter checkpoint加载权重
        
        Args:
            editor_ckpt_path: 编辑器检查点路径
            inverter_ckpt_path: 反转器检查点路径
        """
        print("=" * 60)
        print("推理模式：分别加载编辑器和反转器权重")
        print("=" * 60)
        
        # 1. 加载编辑器检查点中的判别器权重
        print(f"从编辑器检查点加载判别器权重: {editor_ckpt_path}")
        editor_ckpt = torch.load(editor_ckpt_path, map_location="cpu")
        self.load_disc_from_ckpt(editor_ckpt)
        
        # 2. 加载编辑器检查点中的编码器权重
        print(f"从编辑器检查点加载编码器权重: {editor_ckpt_path}")
        encoder_state_dict = get_keys_with_prefix_handling(editor_ckpt, "encoder")
        try:
            self.encoder.load_state_dict(encoder_state_dict, strict=True)
            print("成功加载编码器权重")
        except Exception as e:
            print(f"加载编码器权重时出错: {e}")
            print("尝试非严格模式加载...")
            self.encoder.load_state_dict(encoder_state_dict, strict=False)
        
        # 3. 加载反转器检查点中的反转器权重
        print(f"从反转器检查点加载反转器权重: {inverter_ckpt_path}")
        inverter_ckpt = torch.load(inverter_ckpt_path, map_location="cpu")
        inverter_state_dict = get_keys_with_prefix_handling(inverter_ckpt, "inverter")
        try:
            self.inverter.load_state_dict(inverter_state_dict, strict=True)
            print("成功加载反转器权重")
        except Exception as e:
            print(f"加载反转器权重时出错: {e}")
            print("尝试非严格模式加载...")
            self.inverter.load_state_dict(inverter_state_dict, strict=False)
        
        # 4. 加载StyleGAN解码器权重
        print(f"加载StyleGAN解码器权重: {self.opts.stylegan_weights}")
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        
        # 5. 设置模型为评估模式
        self.inverter = self.inverter.eval().to(self.device)
        self.decoder = self.decoder.eval().to(self.device)
        self.e4e_encoder = self.e4e_encoder.to(self.device)
        
        # 6. 冻结不需要梯度的组件
        toogle_grad(self.inverter, False)
        toogle_grad(self.decoder, False)
        toogle_grad(self.e4e_encoder, False)
        
        print("=" * 60)
        print("权重加载完成")
        print("=" * 60)

    def load_weights(self):
        """
        训练时的权重加载方法
        保持原有逻辑不变
        """
        if self.opts.checkpoint_path != "":
            print(f"Loading from checkpoint: {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            
            # 使用修正后的函数调用方式
            encoder_state_dict = get_keys_with_prefix_handling(ckpt, "encoder")
            inverter_state_dict = get_keys_with_prefix_handling(ckpt, "inverter")
            
            try:
                self.encoder.load_state_dict(encoder_state_dict, strict=True)
                print("成功加载编码器权重")
            except Exception as e:
                print(f"加载编码器权重时出错: {e}")
                print("尝试非严格模式加载...")
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
            
            # ckpt = torch.load(self.inverter_pth, map_location="cpu")
            # self.inverter.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            try:
                self.inverter.load_state_dict(inverter_state_dict, strict=True)
                print("成功加载反转器权重")
            except Exception as e:
                print(f"加载反转器权重时出错: {e}")
                print("尝试非严格模式加载...")
                self.inverter.load_state_dict(inverter_state_dict, strict=False)
        else:
            print(f"Loading Discriminator and Inverter from Inverter checkpoint: {self.inverter_pth}")
            ckpt = torch.load(self.inverter_pth, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.inverter.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

        self.inverter = self.inverter.eval().to(self.device)
        toogle_grad(self.inverter, False)

        print("Loading Decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        self.decoder = self.decoder.eval().to(self.device)
        toogle_grad(self.decoder, False)

        # 不加载E4E预训练模型
        print("Skip loading E4E pretrained model")
        self.e4e_encoder = self.e4e_encoder.to(self.device)
        toogle_grad(self.e4e_encoder, False)

        # 不加载编码器权重，从头训练
        print("Training Encoder from scratch")

    def set_encoder(self):
        self.inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18) 
        self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        feat_editor = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        return feat_editor  # trainable part
    
    def forward(self, x, return_latents=False, n_iter=1e5):
        # 确保latent_avg在正确的设备上
        if self.latent_avg.device != x.device:
            self.latent_avg = self.latent_avg.to(x.device)
            
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            w_recon, predicted_feat = self.inverter.fs_backbone(x)
            w_recon = w_recon + self.latent_avg
                    
            _, w_feats = self.decoder(
                [w_recon],
                input_is_latent=True,
                return_features=True,
                is_stylespace=False,
                randomize_noise=False,
                early_stop=64
            )

            w_feat = w_feats[9]  # bs x 512 x 64 x 64 
            
            fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
            delta = torch.zeros_like(fused_feat)  # inversion case

        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )

        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                predicted_feat = predicted_feat.cpu()
            return images, w_recon, fused_feat, predicted_feat
        return images


@methods_registry.add_to_registry("fse_inverter", stop_args=("self", "checkpoint_path"))
class FSEInverter(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None):
        super(FSEInverter, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.encoder = self.set_encoder()

        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None
        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


    def load_disc(self):
        print("Loading default Discriminator from ", self.opts.stylegan_weights_pkl)
        # We used the hyperinverter discriminator since it has a cars checkpoint
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)

    def load_disc_from_ckpt(self, ckpt):
        # 检查是否有module.discriminator前缀的键
        has_module_prefix = any(key.startswith("module.discriminator.") for key in ckpt["state_dict"].keys())
        
        # 检查是否有discriminator前缀的键
        unique_keys = set(key.split(".")[0] for key in ckpt["state_dict"].keys())
        has_disc_prefix = "discriminator" in unique_keys
        
        # 情况1: 有module.discriminator前缀
        if has_module_prefix:
            print("检测到module.discriminator前缀，使用前缀处理方式加载判别器")
            disc_state_dict = get_keys_with_prefix_handling(ckpt, "discriminator")
            try:
                self.discriminator.load_state_dict(disc_state_dict, strict=True)
                print("成功加载判别器权重")
            except Exception as e:
                print(f"加载判别器权重时出错: {e}")
                print("尝试非严格模式加载...")
                self.discriminator.load_state_dict(disc_state_dict, strict=False)
        
        # 情况2: 有discriminator前缀（原始方式）
        elif has_disc_prefix:
            print("检测到discriminator前缀，使用原始方式加载判别器")
            try:
                self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
                print("成功加载判别器权重")
            except Exception as e:
                print(f"加载判别器权重时出错: {e}")
                print("尝试非严格模式加载...")
                self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=False)
        
        # 情况3: 两种前缀都没有找到
        else:
            print("未找到判别器权重，保留默认权重")

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print("Loading from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            
            # 使用修正后的函数调用方式处理encoder权重
            encoder_state_dict = get_keys_with_prefix_handling(ckpt, "encoder")
            try:
                self.encoder.load_state_dict(encoder_state_dict, strict=True)
                print("成功加载编码器权重")
            except Exception as e:
                print(f"加载编码器权重时出错: {e}")
                print("尝试非严格模式加载...")
                self.encoder.load_state_dict(encoder_state_dict, strict=False)

        print("Loading decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)

    def set_encoder(self):
        inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18)
        return inverter  # trainable part
    
    def forward(self, x, return_latents=False, n_iter=1e5):
        # 确保latent_avg在正确的设备上
        if self.latent_avg.device != x.device:
            self.latent_avg = self.latent_avg.to(x.device)
            
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        w_recon, predicted_feat = self.encoder.fs_backbone(x)
        w_recon = w_recon + self.latent_avg
                
        _, w_feats = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )

        w_feat = w_feats[9]  # bs x 512 x 64 x 64 
        fused_feat = self.encoder.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        feats = [None] * 9 + [fused_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )
        
        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                w_feat = w_feat.cpu()
            return images, w_recon, fused_feat, w_feat
        return images


@methods_registry.add_to_registry("vivface_fse", stop_args=("self", "checkpoint_path"))
class VIVFaceFSE(nn.Module):
    """
    ViVFace身份与表情解耦模型
    
    该模型基于StyleFeatureEditor架构，但使用了双路径（w_latent和ss_latent）处理身份和表情信息。
    - w_latent: 控制身份特征
    - ss_latent: 控制表情特征
    """
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None,
                 e4e_path=None,
                 progressive_stage=None,
                 ss_style_count=10):
        super(VIVFaceFSE, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024,
            "ss_styles": ss_style_count
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.e4e_path = e4e_path or self.opts.e4e_path

        # 创建编码器和解码器
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None
        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()
        
        # 设置训练阶段
        if progressive_stage is not None:
            if isinstance(progressive_stage, str):
                progressive_stage = getattr(ProgressiveStage, progressive_stage)
            self.encoder.set_progressive_stage(progressive_stage)

    def load_disc(self):
        print("Loading default Discriminator from ", self.opts.stylegan_weights_pkl)
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)

    def load_disc_from_ckpt(self, ckpt):
        unique_keys = set(key.split(".")[0] for key in ckpt["state_dict"].keys())
        if "discriminator" in unique_keys:
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
        else:
            print("Can not find Discriminator weights in checkpoint, leave default weights.")

    def load_weights(self):
        # 检查是否有检查点文件路径
        if self.opts.checkpoint_path and self.opts.checkpoint_path != "":
            # 模式1：从检查点继续训练（加载所有组件）
            print(f"继续训练模式：加载完整检查点 {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            
            # 加载状态字典
            if "state_dict" in ckpt:
                # 处理带module前缀的键（DDP模式保存的模型）
                state_dict = {}
                for key, val in ckpt["state_dict"].items():
                    if key.startswith("module."):
                        # 移除"module."前缀
                        state_dict[key[7:]] = val
                    else:
                        state_dict[key] = val
                
                # 加载编码器权重
                encoder_dict = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
                if encoder_dict:
                    # 移除"encoder."前缀
                    encoder_dict = {k[8:]: v for k, v in encoder_dict.items()}
                    self.encoder.load_state_dict(encoder_dict, strict=True)
                    print("成功加载编码器权重")
                else:
                    print("警告：检查点中没有找到编码器权重")
                
                # 加载判别器权重
                self.load_disc_from_ckpt(ckpt)
            
            # 从检查点加载latent_avg
            if "latent_avg" in ckpt:
                self.latent_avg = ckpt["latent_avg"].to(self.device)
                print("成功加载latent_avg")
        else:
            # 模式2：新训练（只加载StyleGAN生成器和判别器，编码器从头训练）
            print("新训练模式：加载预训练StyleGAN，编码器从头训练")
            
            # 加载StyleGAN解码器
            try:
                print("加载StyleGAN解码器：", self.opts.stylegan_weights)
                ckpt = torch.load(self.opts.stylegan_weights)
                self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
                self.latent_avg = ckpt['latent_avg'].to(self.device)
                print("成功加载StyleGAN2解码器")
            except Exception as e:
                print(f"加载StyleGAN2解码器失败: {e}")
                # 确保在加载失败时也有一个latent_avg
                self.latent_avg = torch.zeros(512).to(self.device)
            
            print("编码器将从头训练")
        
        # 设置解码器为评估模式并冻结参数
        self.decoder = self.decoder.eval().to(self.device)
        toogle_grad(self.decoder, False)

    def set_encoder(self):
        # ViVFace使用E4E编码器，该编码器已添加了ss_latent输出功能
        encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        return encoder  # 可训练部分
    
    def forward(self, x, return_latents=False, randomize_noise=True, w_latent=None, ss_latent=None):
        """
        VIVFaceFSE的前向传播函数
        
        Args:
            x: 输入图像
            return_latents: 是否返回潜在编码
            randomize_noise: 是否在生成过程中使用随机噪声，默认为True，与原始ViVFace保持一致
            w_latent: 可选，直接提供的身份编码
            ss_latent: 可选，直接提供的表情编码
        """
        # 确保latent_avg在正确的设备上
        if self.latent_avg is not None and x.device != self.latent_avg.device:
            self.latent_avg = self.latent_avg.to(x.device)
            
        # 调整图像尺寸
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        # 如果没有提供编码，使用编码器生成
        if w_latent is None or ss_latent is None:
            # 使用E4E编码器生成w_latent和ss_latent
            w_latent, ss_latent = self.encoder(x)
            
            # 添加平均潜在编码
            if self.latent_avg is not None:
                w_latent = w_latent + self.latent_avg.unsqueeze(0).repeat(w_latent.shape[0], 1, 1)
        
        # 使用双路径生成图像
        images, return_dict = self.decoder(
            [w_latent],
            input_is_latent=True,
            return_latents=True,
            randomize_noise=randomize_noise,
            ss_latent=ss_latent
        )

        del return_dict

        torch.cuda.empty_cache()

        if return_latents:
            # 返回图像和两路径的潜在编码
            return images, w_latent, ss_latent
        return images
        
    def generate_from_latents(self, w_latent, ss_latent, randomize_noise=False):
        """
        直接从潜在编码生成图像，用于内存优化
        
        Args:
            w_latent: 身份编码，形状为[batch_size, 18, 512]
            ss_latent: 表情编码，形状为[batch_size, 18, 512]
            randomize_noise: 是否使用随机噪声，默认为False
            
        Returns:
            生成的图像
        """
        # 使用双路径生成图像
        images, _ = self.decoder(
            [w_latent],
            input_is_latent=True,
            return_latents=False,
            randomize_noise=randomize_noise,
            ss_latent=ss_latent
        )
        
        # 立即清理缓存
        torch.cuda.empty_cache()
        
        return images
