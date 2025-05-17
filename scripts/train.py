import argparse
import os
import random
import sys
import torch
import json
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

sys.path = ['.'] + sys.path

from arguments import training_arguments
from runners.training_runners import training_runners
from utils.common_utils import printer, setup_seed

def setup_ddp(rank, world_size, port):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

def run_training(rank, world_size, config_file_path, conf_cli_path=None):
    """在单个进程中运行训练"""
    # 加载基本配置
    config = OmegaConf.load(config_file_path)
    
    # 如果有命令行参数，也加载它们
    if conf_cli_path and os.path.exists(conf_cli_path):
        conf_cli = OmegaConf.load(conf_cli_path)
        config = OmegaConf.merge(config, conf_cli)
    
    # 设置DDP环境
    if config.dist.enabled:
        setup_ddp(rank, world_size, config.dist.port)
        # 设置当前进程的rank和总进程数
        config.dist.rank = rank
        config.dist.world_size = world_size
        
    # 设置随机种子（每个进程不同）
    if config.dist.enabled:
        setup_seed(config.exp.seed + rank)
    else:
        setup_seed(config.exp.seed)
    
    # 仅在主进程打印配置
    if not config.dist.enabled or config.dist.rank == 0:
        printer(config)
    
    # 初始化训练器并运行
    trainer = training_runners[config.train.train_runner](config)
    trainer.setup()
    trainer.run()
    
    # 清理DDP环境
    if config.dist.enabled:
        cleanup_ddp()

if __name__ == "__main__":
    # 加载配置
    config = training_arguments.load_config()
    
    # 检查是否启用分布式训练
    if config.dist.enabled:
        # 获取可用GPU数量
        world_size = torch.cuda.device_count()
        if hasattr(config.dist, 'world_size') and config.dist.world_size > 0:
            world_size = min(world_size, config.dist.world_size)
            
        if world_size > 1:
            print(f"Starting distributed training with {world_size} GPUs")
            
            # 保存配置到临时文件
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                config_path = tmp.name
                OmegaConf.save(config=config, f=config_path)
            
            # 保存命令行参数到临时文件
            conf_cli = OmegaConf.from_cli()
            cli_path = None
            if conf_cli:
                with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                    cli_path = tmp.name
                    OmegaConf.save(config=conf_cli, f=cli_path)
            
            try:
                # 启动多进程
                mp.spawn(
                    run_training,
                        args=(world_size, config_path, cli_path),
                    nprocs=world_size,
                    join=True
                )
            finally:
                # 清理临时文件
                if os.path.exists(config_path):
                    os.unlink(config_path)
                if cli_path and os.path.exists(cli_path):
                    os.unlink(cli_path)
        else:
            print("Warning: Distributed training enabled but only 1 GPU available. Falling back to single GPU.")
            config.dist.enabled = False
            
            # 保存配置到临时文件
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                config_path = tmp.name
                OmegaConf.save(config=config, f=config_path)
                
            # 保存命令行参数到临时文件
            conf_cli = OmegaConf.from_cli()
            cli_path = None
            if conf_cli:
                with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                    cli_path = tmp.name
                    OmegaConf.save(config=conf_cli, f=cli_path)
            
            try:
                run_training(0, 1, config_path, cli_path)
            finally:
                # 清理临时文件
                if os.path.exists(config_path):
                    os.unlink(config_path)
                if cli_path and os.path.exists(cli_path):
                    os.unlink(cli_path)
    else:
        # 单GPU训练
        # 保存配置到临时文件
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            config_path = tmp.name
            OmegaConf.save(config=config, f=config_path)
        
        # 保存命令行参数到临时文件
        conf_cli = OmegaConf.from_cli()
        cli_path = None
        if conf_cli:
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                cli_path = tmp.name
                OmegaConf.save(config=conf_cli, f=cli_path)
        
        try:
            run_training(0, 1, config_path, cli_path)
        finally:
            # 清理临时文件
            if os.path.exists(config_path):
                os.unlink(config_path)
            if cli_path and os.path.exists(cli_path):
                os.unlink(cli_path)
