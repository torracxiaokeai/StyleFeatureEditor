import torch
import collections
import logging
import omegaconf
import wandb
import datetime
import glob
import os
import json

from PIL import Image


class BaseTimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def stop(self):
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end) / 1000

class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end) / 1000
        if self.info:
            self.info[f"duration/{self.log_event}"] = self.duration


class _StreamingMean:
    def __init__(self, val=None, counts=None):
        if val is None:
            self.mean = 0.0
            self.counts = 0
        else:
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy()
            self.mean = val
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1

    def update(self, mean, counts=1):
        if isinstance(mean, torch.Tensor):
            mean = mean.data.cpu().numpy()
        elif isinstance(mean, _StreamingMean):
            mean, counts = mean.mean, mean.counts * counts
        assert counts >= 0
        if counts == 0:
            return
        total = self.counts + counts
        self.mean = self.counts / total * self.mean + counts / total * mean
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.mean, self.counts)
        if isinstance(other, _StreamingMean):
            if other.counts == 0:
                return new
            else:
                new.update(other.mean, other.counts)
        else:
            new.update(other)
        return new


class StreamingMeans(collections.defaultdict):
    def __init__(self):
        super().__init__(_StreamingMean)

    def __setitem__(self, key, value):
        if isinstance(value, _StreamingMean):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, _StreamingMean(value))

    def update(self, *args, **kwargs):
        for_update = dict(*args, **kwargs)
        for k, v in for_update.items():
            self[k].update(v)

    def to_dict(self, prefix=""):
        return dict((prefix + k, v.mean) for k, v in self.items())

    def to_str(self):
        return ", ".join([f"{k} = {v:.3f}" for k, v in self.to_dict().items()])


class ConsoleLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter(
            "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        self.logger.propagate = False

    @staticmethod
    def format_info(info):
        if not info:
            return str(info)
        log_groups = collections.defaultdict(dict)
        for k, v in info.to_dict().items():
            prefix, suffix = k.split("/", 1)
            log_groups[prefix][suffix] = f"{v:.3f}" if isinstance(v, float) else str(v)
        formatted_info = ""
        max_group_size = len(max(log_groups, key=len)) + 2
        max_k_size = max([len(max(g, key=len)) for g in log_groups.values()]) + 1
        max_v_size = (
            max([len(max(g.values(), key=len)) for g in log_groups.values()]) + 1
        )
        for group, group_info in log_groups.items():
            group_str = [
                f"{k:<{max_k_size}}={v:>{max_v_size}}" for k, v in group_info.items()
            ]
            max_g_size = len(max(group_str, key=len)) + 2
            group_str = "".join([f"{g:>{max_g_size}}" for g in group_str])
            formatted_info += f"\n{group + ':':<{max_group_size}}{group_str}"
        return formatted_info

    def log_iter(self, epoch_num, iter_num, num_iters, iter_info, event="epoch"):
        output_info = f"{event.upper()} {epoch_num}, ITER {iter_num}/{num_iters}:"
        output_info += self.format_info(iter_info)
        self.logger.info(output_info)

    def log_epoch(self, epoch_info, epoch_num):
        output_info = f"EPOCH {epoch_num}:"
        output_info += self.format_info(epoch_info)
        self.logger.info(output_info)


class WandbLogger:
    def __init__(self, config):
        # 尝试多种方式获取 wandb API key
        api_key = None
        
        # 方法1: 检查自定义的 WANDB_KEY 环境变量
        if 'WANDB_KEY' in os.environ:
            api_key = os.environ['WANDB_KEY'].strip()
        # 方法2: 检查标准的 WANDB_API_KEY 环境变量
        elif 'WANDB_API_KEY' in os.environ:
            api_key = os.environ['WANDB_API_KEY'].strip()
        # 方法3: 尝试使用已经登录的 wandb 状态
        else:
            try:
                # 检查是否已经有有效的登录状态
                import wandb.sdk.lib.apikey
                if wandb.api.api_key:
                    api_key = wandb.api.api_key
                else:
                    print("Warning: No wandb API key found. Trying to use existing login...")
            except Exception as e:
                print(f"Warning: Could not access wandb API key: {e}")
        
        # 登录 wandb
        if api_key:
            wandb.login(key=api_key, relogin=True)
        else:
            # 如果没有找到 API key，尝试使用现有的登录状态
            try:
                wandb.login(relogin=False)
            except Exception as e:
                print(f"Error: Could not login to wandb: {e}")
                raise RuntimeError("Please set WANDB_KEY environment variable or run 'wandb login' first")
        
        if config.train.resume_path == "":
            config_for_logger = omegaconf.OmegaConf.to_container(config)
            self.wandb_args = {
                "id": wandb.util.generate_id(),
                "project": config.exp.wandb_project,
                "name": config.exp.name,
                "config": config_for_logger,
            }
            wandb.init(**self.wandb_args, resume="allow")

            run_dir = wandb.run.dir
            print("run_dir", run_dir)

            code = wandb.Artifact("project-source", type="code")
            for path in glob.glob("**/*.py", recursive=True):
                if not path.startswith("wandb"):
                    if os.path.basename(path) != path:
                        code.add_dir(
                            os.path.dirname(path), name=os.path.dirname(path)
                        )
                    else:
                        code.add_file(os.path.basename(path), name=path)
            wandb.run.log_artifact(code)
        else:
            print(f"Resume training from {config.train.resume_path}")
            with open(config.train.resume_path, "r") as f:
                options = json.load(f)

            self.wandb_args = {
                "id": options['id'],
                "project": options['project'],
                "name": options['name'],
                "config": options['config'],
            }
            wandb.init(resume=True, **self.wandb_args)

    @staticmethod
    def log_epoch(iter_info, step):
        wandb.log(
                data={k: v.mean for k, v in iter_info.items()},
                step=step + 1,
                commit=True,
            )

    @staticmethod
    def log_special_pics(pics, captions, paths):
        to_log = {}
        
        # 添加长度检查，避免索引越界
        max_pics = len(pics)
        max_paths = len(paths)
        
        if max_pics == 0:
            print("Warning: No pictures to log")
            return
            
        if max_pics != max_paths:
            print(f"Warning: Mismatch between pics count ({max_pics}) and paths count ({max_paths})")
            # 使用较小的数量以避免越界
            actual_count = min(max_pics, max_paths)
        else:
            actual_count = max_pics
            
        for i in range(actual_count):
            path = paths[i] if i < max_paths else f"img_{i}"
            pic = pics[i] if i < max_pics else pics[0]  # 如果图片不够，重复使用第一张
            caption = captions.get(path, f"Image {i}") if captions else f"Image {i}"
            
            try:
                to_log[path] = wandb.Image(pic, caption=caption)
            except Exception as e:
                print(f"Error logging image {i} for path {path}: {e}")
                continue
                
        if to_log:  # 只有在有内容时才记录
            wandb.log(to_log)


class BlankWandbLogger:
    def __init__(self):
        self.wandb_args = None

    @staticmethod
    def log_epoch(*args, **kwargs):
        pass

    @staticmethod  
    def log_special_pics(*args, **kwargs):
        pass


class TrainigLogger:
    def __init__(self, config):
        self.console_logger = ConsoleLogger("")
        self.config = config
        
        # 判断是否为主进程或非分布式环境
        self.is_main_process = not hasattr(config, 'dist') or not config.dist.enabled or config.dist.rank == 0
        
        if config.exp.wandb == True and self.is_main_process:
            self.wandb_logger = WandbLogger(config)
        else:
            self.wandb_logger = BlankWandbLogger()

        self.trainig_steps = config.train.steps 
        self.val_step = config.train.val_step

    def log_train_time_left(self, iter_info, step):
        # 只在主进程记录训练时间
        if not self.is_main_process:
            return
            
        float_iter_time = iter_info["duration/iter_train"].mean
        float_val_time = iter_info["duration/iter_val"].mean
        time_left = str(
            datetime.datetime.fromtimestamp(
                float_iter_time * (self.trainig_steps - step)
                + float_val_time
                * (
                    (self.trainig_steps - step) // self.val_step
                )
            )
            - datetime.datetime.fromtimestamp(0)
        )

        print()
        print(f"Step {step}/{self.trainig_steps}")
        print(f"Time left: {time_left}")
        print(f"Time per step: {iter_info['duration/iter_train'].mean :.3f}")
        print()
        print()

    def save_train_logs(self, iter_info, step):
        # 只在主进程保存日志
        if not self.is_main_process:
            return
            
        self.wandb_logger.log_epoch(iter_info, step)
        self.console_logger.log_epoch(iter_info, step)

        self.log_train_time_left(iter_info, step)

    def save_validation_logs(self, orig_pics, method_pics, captions, special_paths):
        # 只在主进程保存验证日志
        if not self.is_main_process:
            return
        
        # 检查输入数据的一致性
        if len(orig_pics) != len(method_pics):
            print(f"Warning: Mismatch between orig_pics ({len(orig_pics)}) and method_pics ({len(method_pics)})")
            min_count = min(len(orig_pics), len(method_pics))
            orig_pics = orig_pics[:min_count]
            method_pics = method_pics[:min_count]
            
        log_pics = []
        for real_img, fake_img in zip(orig_pics, method_pics):
            concat_img = Image.new(
                "RGB", (real_img.width + fake_img.width, real_img.height)
            )
            concat_img.paste(real_img, (0, 0))
            concat_img.paste(fake_img, (real_img.width, 0))
            log_pics.append(concat_img)

        # 确保special_paths的数量与log_pics匹配
        if len(special_paths) > len(log_pics):
            print(f"Trimming special_paths from {len(special_paths)} to {len(log_pics)}")
            actual_paths = special_paths[:len(log_pics)]
        elif len(special_paths) < len(log_pics):
            print(f"Extending special_paths from {len(special_paths)} to {len(log_pics)}")
            actual_paths = special_paths + [f"img_{i}" for i in range(len(special_paths), len(log_pics))]
        else:
            actual_paths = special_paths

        self.wandb_logger.log_special_pics(log_pics, captions, actual_paths)
