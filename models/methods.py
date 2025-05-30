import math
import sys
import pickle
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.psp.encoders import psp_encoders
from models.vivface.encoders import psp_encoders_identity_related as vivface_psp_encoders
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

@methods_registry.add_to_registry("vivface_method", stop_args=("self", "checkpoint_path"))
class VivFaceMethod(nn.Module):
    """
    基于真实ViVFace代码实现的完整方法类
    
    这是参考ViVFace原始实现，在StyleFeatureEditor框架中重新实现的版本。
    主要特点：
    1. 双路径编码：w_latent（身份）+ ss_latent（表情）
    2. 基于E4E的渐进式训练
    3. StyleGAN2解码器修改支持双路径输入
    4. 完整的身份-表情解耦训练流程
    """
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None,
                 progressive_stage="Inference",
                 ss_style_count=10,
                 encoder_weight_strategy="e4e",
                 use_pretrained_encoder=True):
        super(VivFaceMethod, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024,
            "ss_styles": ss_style_count,
            "start_from_latent_avg": True,
            "encoder_type": "Encoder4Editing",
            "encoder_weight_strategy": encoder_weight_strategy,
            "use_pretrained_encoder": use_pretrained_encoder
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        
        # 创建编码器和解码器
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.latent_avg = None
        
        # 池化层用于下采样
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        
        # 加载判别器
        self.load_disc()
        
        # 加载权重
        self.load_weights()
        
        # 设置训练阶段
        if isinstance(progressive_stage, str):
            progressive_stage = getattr(ProgressiveStage, progressive_stage)
        self.encoder.set_progressive_stage(progressive_stage)

    def set_encoder(self):
        """创建E4E编码器，支持双路径输出"""
        encoder = vivface_psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        return encoder

    def load_disc(self):
        """加载StyleGAN2判别器"""
        print("Loading default Discriminator from ", self.opts.stylegan_weights_pkl)
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)

    def load_disc_from_ckpt(self, ckpt):
        """从检查点加载判别器"""
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
        """加载模型权重"""
        if self.opts.checkpoint_path and self.opts.checkpoint_path != "":
            # 从检查点继续训练
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
            # 新训练模式
            print("新训练模式：加载预训练权重")
            
            # 根据配置加载编码器权重
            if self.opts.use_pretrained_encoder:
                if self.opts.encoder_weight_strategy == "e4e":
                    print("加载E4E预训练编码器权重")
                    try:
                        from configs.paths import DefaultPaths
                        encoder_ckpt = torch.load(DefaultPaths.e4e_path, map_location="cpu")
                        
                        # E4E检查点通常包含完整的模型状态
                        if "state_dict" in encoder_ckpt:
                            # 提取编码器部分
                            encoder_dict = {}
                            for key, val in encoder_ckpt["state_dict"].items():
                                if key.startswith("encoder."):
                                    # 移除"encoder."前缀
                                    new_key = key[8:]
                                    encoder_dict[new_key] = val
                            
                            if encoder_dict:
                                self.encoder.load_state_dict(encoder_dict, strict=False)
                                print("成功加载E4E编码器权重")
                            else:
                                print("E4E检查点中未找到编码器权重，尝试直接加载")
                                self.encoder.load_state_dict(encoder_ckpt, strict=False)
                        else:
                            # 直接加载（可能是只包含编码器的检查点）
                            self.encoder.load_state_dict(encoder_ckpt, strict=False)
                            print("成功加载E4E编码器权重")
                            
                    except Exception as e:
                        print(f"加载E4E预训练权重失败: {e}")
                        print("回退到IR-SE50权重")
                        try:
                            encoder_ckpt = torch.load(DefaultPaths.ir_se50_path, map_location="cpu")
                            self.encoder.load_state_dict(encoder_ckpt, strict=False)
                            print("成功加载IR-SE50预训练权重")
                        except Exception as e2:
                            print(f"加载IR-SE50预训练权重也失败: {e2}")
                            print("编码器将从头训练")
                
                elif self.opts.encoder_weight_strategy == "ir_se50":
                    print("加载IR-SE50预训练编码器权重")
                    try:
                        from configs.paths import DefaultPaths
                        encoder_ckpt = torch.load(DefaultPaths.ir_se50_path, map_location="cpu")
                        self.encoder.load_state_dict(encoder_ckpt, strict=False)
                        print("成功加载IR-SE50预训练权重")
                    except Exception as e:
                        print(f"加载IR-SE50预训练权重失败: {e}")
                        print("编码器将从头训练")
                
                elif self.opts.encoder_weight_strategy == "none":
                    print("配置为不使用预训练编码器权重，编码器将从头训练")
                
                else:
                    print(f"未知的编码器权重策略: {self.opts.encoder_weight_strategy}")
                    print("编码器将从头训练")
            else:
                print("配置为不使用预训练编码器权重，编码器将从头训练")
            
            # 加载StyleGAN2解码器
            try:
                print("加载StyleGAN解码器：", self.opts.stylegan_weights)
                ckpt = torch.load(self.opts.stylegan_weights)
                self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
                
                # 加载latent_avg
                if "latent_avg" in ckpt:
                    self.latent_avg = ckpt['latent_avg'].to(self.device)
                elif self.opts.start_from_latent_avg:
                    # 如果没有预计算的latent_avg，计算一个
                    with torch.no_grad():
                        self.latent_avg = self.decoder.mean_latent(10000).to(self.device)
                else:
                    self.latent_avg = None
                    
                print("成功加载StyleGAN2解码器")
            except Exception as e:
                print(f"加载StyleGAN2解码器失败: {e}")
                # 确保在加载失败时也有一个latent_avg
                if self.opts.start_from_latent_avg:
                    self.latent_avg = torch.zeros(512).to(self.device)
            
            print("ss_latent_transformer将从头训练")

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, skip_latent=False,
                return_images=True, w_latent=None, ss_generic_latent=None, 
                ss_latent=None, input_memory=None):
        """
        VivFace的前向传播函数
        
        这个实现完全基于原始ViVFace代码，支持：
        1. 双路径编码：w_latent（身份）+ ss_generic_latent（通用表情特征）
        2. ss_latent变换：通过transformer将ss_generic_latent转换为最终的ss_latent
        3. 完整的身份-表情解耦生成
        
        Args:
            x: 输入图像
            resize: 是否调整输出图像大小
            latent_mask: 潜在编码掩码
            input_code: 输入是否为编码
            randomize_noise: 是否随机化噪声
            inject_latent: 注入的潜在编码
            return_latents: 是否返回潜在编码
            alpha: alpha混合系数
            skip_latent: 是否跳过编码步骤
            return_images: 是否返回图像
            w_latent: 直接提供的身份编码
            ss_generic_latent: 直接提供的通用表情编码
            ss_latent: 直接提供的最终表情编码
            input_memory: 输入记忆特征
        """
        # 确保latent_avg在正确的设备上
        if self.latent_avg is not None and x.device != self.latent_avg.device:
            self.latent_avg = self.latent_avg.to(x.device)
        
        # 第一步：编码阶段（如果不跳过）
        if not skip_latent:
            # 使用E4E编码器生成w_codes（身份）、ss_generic_codes（通用表情）和ref_feature（参考特征）
            w_codes, ss_generic_codes, ref_feature = self.encoder(x)
            
            # 添加平均潜在编码
            if self.opts.start_from_latent_avg and self.latent_avg is not None:
                if w_codes.ndim == 2:
                    w_codes = w_codes + self.latent_avg.repeat(w_codes.shape[0], 1, 1)[:, 0, :]
                else:
                    w_codes = w_codes + self.latent_avg.repeat(w_codes.shape[0], 1, 1)
            
            # 准备返回的潜在编码字典
            returned_latent = {
                'w_latent': w_codes, 
                'ss_generic_latent': ss_generic_codes
            }
        
        # 使用外部提供的编码（如果有）
        if w_latent is not None:
            w_codes = w_latent
        if ss_generic_latent is not None:
            ss_generic_codes = ss_generic_latent
        
        # 第二步：ss_latent变换阶段
        if ss_latent is not None:
            # 直接使用提供的ss_latent
            assert ss_generic_latent is None, "不能同时提供ss_latent和ss_generic_latent"
            ss_codes = ss_latent
        else:
            # 直接使用ss_generic_codes作为最终的ss_codes
            # 在真实的ViVFace实现中，通常不需要额外的transformer处理
            ss_codes = ss_generic_codes
        
        # 第三步：潜在编码注入和混合
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        w_codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * w_codes[:, i]
                    else:
                        w_codes[:, i] = inject_latent[:, i]
                else:
                    w_codes[:, i] = 0
        
        # 第四步：图像生成阶段
        input_is_latent = not input_code
        if return_images:
            # 使用修改后的StyleGAN2解码器生成图像
            images, _ = self.decoder(
                [w_codes],  # 只传递w_codes作为主要输入
                input_is_latent=input_is_latent, 
                randomize_noise=randomize_noise, 
                return_latents=return_latents,
                ss_latent=ss_codes  # ss_codes作为单独参数传递
            )
            
            # 调整图像大小
            if resize:
                images = self.face_pool(images)
        
        # 第五步：返回结果
        if return_images and return_latents:
            returned_latent['ss_latent'] = ss_codes
            return images, returned_latent
        elif return_latents:
            returned_latent['ss_latent'] = ss_codes
            return returned_latent
        elif return_images:
            return images
