#!/usr/bin/env python3
"""
FSE Inverter 仅反演脚本
用于调用 FSEInverterInferenceRunner 只进行图像反演，不进行编辑
"""

import sys
import torch
import os
from pathlib import Path
import argparse

# 添加项目根目录到Python路径
sys.path = ['.'] + sys.path

from arguments import inference_arguments
from runners.inference_runners import inference_runner_registry
from utils.common_utils import printer, setup_seed
from omegaconf import OmegaConf


def create_inversion_only_config(
    input_dir,
    output_dir,
    checkpoint_path,
    device="0",
    batch_size=4
):
    """
    创建仅反演的配置对象
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        checkpoint_path: 模型检查点路径
        device: 设备ID
        batch_size: 批大小
    
    Returns:
        配置对象
    """
    config = {
        "exp": {
            "output_dir": output_dir,
            "seed": 42,
            "domain": "human_faces"
        },
        "inference": {
            "inference_runner": "fse_inverter_inference_runner",
            "editings_data": {}  # 空的编辑数据，不进行任何编辑
        },
        "data": {
            "inference_dir": input_dir,
            "transform": "face_1024"
        },
        "model": {
            "method": "fse_inverter",
            "device": device,
            "batch_size": batch_size,
            "workers": 4,
            "checkpoint_path": checkpoint_path
        },
        "methods_args": {
            "fse_inverter": {
                "device": f"cuda:{device}" if device.isdigit() else device,
                "paths": {
                    "psp_path": "pretrained_models/psp_ffhq_encode.pt",
                    "e4e_path": "pretrained_models/e4e_ffhq_encode.pt",
                    "farl_path": "pretrained_models/face_parsing.farl.lapa.main_ema_136500_jit191.pt",
                    "mobile_net_pth": "pretrained_models/mobilenet0.25_Final.pth",
                    "ir_se50_path": "pretrained_models/model_ir_se50.pth",
                    "stylegan_weights": "pretrained_models/stylegan2-ffhq-config-f.pt",
                    "stylegan_car_weights": "pretrained_models/stylegan2-car-config-f-new.pkl",
                    "stylegan_weights_pkl": "pretrained_models/stylegan2-ffhq-config-f.pkl",
                    "arcface_model_path": "pretrained_models/iresnet50-7f187506.pth",
                    "moco": "pretrained_models/moco_v2_800ep_pretrain.pt",
                    "curricular_face_path": "pretrained_models/CurricularFace_Backbone.pth",
                    "mtcnn": "pretrained_models/mtcnn",
                    "landmark": "pretrained_models/79999_iter.pth"
                }
            }
        }
    }
    
    return OmegaConf.create(config)


def run_inversion_only(config):
    """
    运行仅反演推理
    
    Args:
        config: 配置对象
    """
    print("=" * 60)
    print("FSE Inverter 仅反演模式")
    print("=" * 60)
    
    # 创建推理运行器
    inference_runner = inference_runner_registry[config.inference.inference_runner](config)
    
    # 设置运行器
    print("正在设置推理运行器...")
    inference_runner.setup()
    
    # 只运行反演，不运行编辑
    print("开始运行图像反演...")
    inference_runner.run_inversion()
    
    print("=" * 60)
    print("图像反演完成！")
    print(f"反演结果保存在: {config.exp.output_dir}/inversion/")
    print("注意: 跳过了编辑步骤")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FSE Inverter 仅反演模式")
    parser.add_argument("--input_dir", "-i", 
                       help="输入图像目录")
    parser.add_argument("--output_dir", "-o", default="results/inversion_only",
                       help="输出目录 (默认: results/inversion_only)")
    parser.add_argument("--checkpoint", "-c", 
                       default="experiments/fse_inverter_train_005/iteration_47000.pt",
                       help="模型检查点路径")
    parser.add_argument("--device", "-d", default="0",
                       help="设备ID (默认: 0)")
    parser.add_argument("--batch_size", "-b", type=int, default=4,
                       help="批大小 (默认: 4)")
    parser.add_argument("--config", 
                       help="使用配置文件而不是命令行参数")
    
    args = parser.parse_args()
    
    if args.config:
        # 使用配置文件模式
        print(f"使用配置文件: {args.config}")
        
        # 临时设置配置文件路径
        sys.argv = ["run_fse_inverter_inversion_only.py", f"exp.config={args.config}"]
        
        # 加载配置
        config = inference_arguments.load_config()
        
        # 设置随机种子
        setup_seed(config.exp.seed)
        
        # 打印配置信息
        printer(config)
        
        # 检查必要的路径
        if not os.path.exists(config.data.inference_dir):
            print(f"错误: 输入图像目录不存在: {config.data.inference_dir}")
            print("请在配置文件中设置正确的 data.inference_dir 路径")
            return
        
        if not os.path.exists(config.model.checkpoint_path):
            print(f"错误: 模型检查点不存在: {config.model.checkpoint_path}")
            print("请在配置文件中设置正确的 model.checkpoint_path 路径")
            return
        
        # 创建输出目录
        output_dir = Path(config.exp.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行仅反演
        run_inversion_only(config)
        
    else:
        # 使用命令行参数模式
        if not args.input_dir:
            print("错误: 必须指定输入目录 (--input_dir)")
            parser.print_help()
            return
        
        print("=" * 60)
        print("FSE Inverter 仅反演模式 (命令行)")
        print("=" * 60)
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"检查点: {args.checkpoint}")
        print(f"设备: {args.device}")
        print(f"批大小: {args.batch_size}")
        print("=" * 60)
        
        # 检查输入目录
        if not os.path.exists(args.input_dir):
            print(f"错误: 输入目录不存在: {args.input_dir}")
            return
        
        # 检查检查点文件
        if not os.path.exists(args.checkpoint):
            print(f"错误: 检查点文件不存在: {args.checkpoint}")
            return
        
        # 创建输出目录
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建配置
        config = create_inversion_only_config(
            args.input_dir, 
            args.output_dir, 
            args.checkpoint, 
            args.device, 
            args.batch_size
        )
        
        # 设置随机种子
        setup_seed(config.exp.seed)
        
        # 运行仅反演
        run_inversion_only(config)


if __name__ == "__main__":
    main() 