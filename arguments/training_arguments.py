import os
from training.losses import disc_losses
from training.optimizers import optimizers
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from omegaconf import OmegaConf, MISSING
from utils.class_registry import ClassRegistry
from models.methods import methods_registry
from metrics.metrics import metrics_registry
import argparse


args = ClassRegistry()


@args.add_to_registry("exp")
@dataclass
class ExperimentArgs:
    config_dir: str = str(Path(__file__).resolve().parent / "configs")
    config: str = MISSING
    exp_dir: str = "experiments"
    name: str = MISSING
    seed: int = 1
    root: str = os.getenv("EXP_ROOT", ".")
    wandb: bool = True
    wandb_project: str = "sfe"
    domain: str = "human_faces"
    resume: bool = False


@args.add_to_registry("data")
@dataclass
class DataArgs:
    special_dir: str = MISSING
    transform: str = "face_1024"
    input_train_dir: str = MISSING
    input_val_dir: str = MISSING
    dataset_type: str = "ffhq"
    source_dir: str = MISSING
    driver_dir: str = MISSING


@args.add_to_registry("train")
@dataclass
class TrainingArgs:
    train_runner: str = "base_training_runner"
    encoder_optimizer: str = "ranger"
    disc_optimizer: str = "adam"
    resume_path: str = ""
    val_metrics: List[str] = field(
        default_factory=lambda: ["msssim", "lpips", "l2", "fid"]
    )
    start_step: int = 0
    steps: int = 300000
    log_step: int = 500
    checkpoint_step: int = 15000
    val_step: int = 15000
    train_dis: bool = False
    dis_train_start_step: int = 150000
    bs_used_before_adv_loss: int = 8
    disc_edits: List[str] = field(
        default_factory=lambda: []
    )
    progressive_stage: str = "WTraining"
    enable_progressive_training: bool = True
    progressive_steps: List[int] = field(
        default_factory=lambda: [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000]
    )
    vivface_checkpoint: str = ""


@args.add_to_registry("model")
@dataclass
class ModelArgs:
    method: str = "fse_full"
    device: str = "0"
    batch_size: int = 4
    workers: int = 4
    checkpoint_path: str = ""


@args.add_to_registry("encoder_losses")
@dataclass
class EncoderLossesArgs:
    l2: float = 0.0
    lpips: float = 0.0
    lpips_scale: float = 0.0
    id: float = 0.0
    moco: float = 0.0
    adv: float = 0.0
    feat_rec: float = 0.0
    feat_rec_l1: float = 0.0
    l2_latent: float = 0.0
    id_vit: float = 0.0
    # ViVFace相关损失函数参数
    vivface_self_rec: float = 0.0
    vivface_reenact: float = 0.0
    vivface_w_consistency: float = 0.0
    vivface_ss_consistency: float = 0.0
    vivface_ss_regularization: float = 0.0
    vivface_id: float = 0.0
    vivface_delta: float = 0.0


@args.add_to_registry("dist")
@dataclass
class DistributedArgs:
    enabled: bool = False
    port: int = 12345
    sync_bn: bool = True
    find_unused_parameters: bool = True
    world_size: int = 4  # -1表示自动检测
    rank: int = 0


MethodsArgs = methods_registry.make_dataclass_from_args("MethodsArgs")
args.add_to_registry("methods_args")(MethodsArgs)

DiscLossesArgs = disc_losses.make_dataclass_from_args("DiscLossesArgs")
args.add_to_registry("disc_losses")(DiscLossesArgs)

OptimizersArgs = optimizers.make_dataclass_from_args("OptimizersArgs")
args.add_to_registry("optimizers")(OptimizersArgs)

MetricsArgs = metrics_registry.make_dataclass_from_args("MetricsArgs")
args.add_to_registry("metrics")(MetricsArgs)


Args = args.make_dataclass_from_classes("Args")


def get_dist_arguments():
    return {
        "enabled": False,
        "port": 12345,
        "sync_bn": True,
        "find_unused_parameters": True,
        "world_size": 4,  # 默认设置为4个GPU
        "rank": 0
    }

def load_config():
    # 创建默认配置
    config = OmegaConf.structured(Args)

    # 解析命令行参数
    conf_cli = OmegaConf.from_cli()
    config.exp.config = conf_cli.exp.config
    config.exp.config_dir = conf_cli.exp.config_dir

    # 加载配置文件
    config_path = os.path.join(config.exp.config_dir, config.exp.config)
    conf_file = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, conf_file)

    # 清理方法参数
    for method in list(config.methods_args.keys()):
        if method != config.model.method:
            config.methods_args.__delattr__(method)

    # 确保分布式训练配置存在
    if not hasattr(config, 'dist'):
        config.dist = OmegaConf.create(get_dist_arguments())

    # 合并命令行参数(最高优先级)
    config = OmegaConf.merge(config, conf_cli)

    return config
