import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import argparse
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import imageio

from runners.inference_runners import FSEInferenceRunner
from datasets.transforms import transforms_registry
from utils.common_utils import tensor2im


class DriverVideoInferenceRunner:
    def __init__(
        self,
        editor_ckpt_pth: str,
        vivface_ckpt_pth: str,
        inverter_ckpt_pth: str = None,
        config_pth: str = "configs/driver_video_inference.yaml",
        source_img_path: str = None,
        driver_dir: str = None,
        output_dir: str = None
    ):
        """
        初始化驱动视频推理运行器
        
        Args:
            editor_ckpt_pth: 编辑器模型检查点路径
            vivface_ckpt_pth: ViVFace模型检查点路径
            inverter_ckpt_pth: 反转器模型检查点路径（可选，如果提供则优先使用）
            config_pth: 配置文件路径
            source_img_path: 源图像路径
            driver_dir: 驱动图像目录
            output_dir: 输出目录
        """
        # 加载配置
        self.config = OmegaConf.load(config_pth)
        
        # 更新配置
        self.config.model.checkpoint_path = editor_ckpt_pth
        self.config.inference.vivface_checkpoint = vivface_ckpt_pth
        
        # 如果提供了源图像和驱动图像目录，则更新配置
        if source_img_path:
            self.config.data.inference_dir = os.path.dirname(source_img_path)
        if driver_dir:
            self.config.inference.driver_dir = driver_dir
        if output_dir:
            self.config.exp.output_dir = output_dir
            
        # 确保方法参数存在
        if 'methods_args' not in self.config:
            self.config.methods_args = {}
        if 'fse_full' not in self.config.methods_args:
            self.config.methods_args.fse_full = {}
        
        # 初始化推理运行器
        self.inference_runner = FSEInferenceRunner(self.config)
        self.inference_runner.setup()
        
        # 使用新的推理权重加载方法
        print("=" * 60)
        print("使用推理模式加载权重")
        print(f"编辑器检查点: {editor_ckpt_pth}")
        print(f"反转器检查点: {inverter_ckpt_pth if inverter_ckpt_pth else editor_ckpt_pth}")
        print("=" * 60)
        
        # 调用新的权重加载方法，优先使用inverter_ckpt_pth
        self.inference_runner.method.load_weights_for_inference(
            editor_ckpt_path=editor_ckpt_pth,
            inverter_ckpt_path=inverter_ckpt_pth if inverter_ckpt_pth else editor_ckpt_pth
        )
        
        self.inference_runner.method.eval()
        self.inference_runner.method.decoder = self.inference_runner.method.decoder.float()
        
        # 设置设备
        self.device = self.inference_runner.device

    def process_video(
        self,
        source_img_path: str,
        driver_dir: str,
        output_dir: str,
        fps: int = 25,
        align_source: bool = False,
        use_mask: bool = False,
        mask_threshold: float = 0.995,
        mask_path: str = None,
    ):
        """
        处理源图像和驱动图像目录，生成编辑后的视频
        
        Args:
            source_img_path: 源图像路径
            driver_dir: 驱动图像目录
            output_dir: 输出目录
            fps: 输出视频的帧率
            align_source: 是否对源图像进行对齐
            use_mask: 是否使用遮罩
            mask_threshold: 遮罩阈值
            mask_path: 自定义遮罩路径
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理源图像
        source_img_path = Path(source_img_path)
        aligned_source_path = source_img_path
        
        if align_source:
            from runners.simple_runner import run_alignment
            aligned_image, unalign_dict = run_alignment(str(source_img_path))
            aligned_source_path = output_path / f"{source_img_path.stem}_aligned.jpg"
            print(f"保存对齐后的源图像到 {aligned_source_path}")
            aligned_image.convert('RGB').save(aligned_source_path)
        
        # 处理遮罩
        mask = None
        if use_mask and mask_path is None:
            from runners.simple_runner import extract_mask
            print("准备遮罩...")
            mask_path = extract_mask(str(aligned_source_path), output_path, trash=mask_threshold)
            print(f"遮罩已保存到 {mask_path}")
        
        if use_mask and mask_path is not None:
            print(f"使用遮罩: {mask_path}")
            mask = Image.open(mask_path).convert("RGB")
            transform = transforms.ToTensor()
            mask = transform(mask).unsqueeze(0).to(self.device)
        
        # 加载源图像
        source_img = Image.open(aligned_source_path).convert("RGB")
        transform_dict = transforms_registry["face_256"]().get_transforms()
        source_tensor = transform_dict["test"](source_img).unsqueeze(0).to(self.device)
        
        # 获取源图像的反演结果
        print("执行源图像反演...")
        inv_images, source_results = self.inference_runner._run_on_batch(source_tensor)
        
        # 保存反演图像
        inv_image = tensor2im(inv_images[0].cpu())
        inv_image_path = output_path / f"{source_img_path.stem}_inversion.jpg"
        inv_image.save(inv_image_path)
        print(f"反演图像已保存到 {inv_image_path}")
        
        # 获取驱动图像列表
        driver_path = Path(driver_dir)
        
        # 检查目录是否存在
        if not driver_path.exists():
            print(f"错误: 驱动图像目录不存在: {driver_dir}")
            return
        
        # 支持更多图像格式
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        driver_files = []
        for ext in image_extensions:
            driver_files.extend(driver_path.glob(ext))
            driver_files.extend(driver_path.glob(ext.upper()))  # 同时查找大写扩展名
        
        if not driver_files:
            print(f"错误: 在目录 {driver_dir} 中未找到支持的图像文件")
            print(f"支持的图像格式: {', '.join(image_extensions)}")
            print("目录内容:")
            for item in driver_path.iterdir():
                print(f"  - {item.name}")
            return
            
        # 按照帧号数字排序
        def get_frame_number(file_path):
            # 从文件名中提取数字
            import re
            numbers = re.findall(r'\d+', file_path.stem)
            if numbers:
                return int(numbers[-1])  # 使用最后一个数字作为帧号
            return 0
            
        driver_files = sorted(driver_files, key=get_frame_number)
        print(f"按顺序排列的驱动帧: {[f.name for f in driver_files]}")
        
        # 创建驱动图像视频
        driver_video_path = output_path / f"driver_frames.mp4"
        print(f"\n创建驱动帧视频: {driver_video_path}")
        
        # 收集驱动帧
        driver_frames = []
        for driver_file in driver_files:
            img = Image.open(driver_file).convert("RGB")
            driver_frames.append(np.array(img))
        
        try:
            # 使用imageio创建驱动视频
            writer = imageio.get_writer(
                str(driver_video_path),
                format='FFMPEG',
                mode='I',
                fps=fps,
                codec='libx264',
                bitrate='5000k',
                quality=None,
                macro_block_size=None,
                output_params=['-pix_fmt', 'yuv420p']
            )
            
            for frame in driver_frames:
                writer.append_data(frame)
            
            writer.close()
            print(f"驱动视频创建成功：{driver_video_path}")
            
        except Exception as e:
            print(f"使用imageio创建驱动视频失败: {str(e)}")
            print("尝试使用OpenCV创建驱动视频...")
            
            if driver_frames:
                height, width, layers = driver_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(str(driver_video_path), fourcc, fps, (width, height))
                
                for frame in driver_frames:
                    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                video.release()
                print(f"使用OpenCV创建驱动视频成功：{driver_video_path}")
        
        print(f"找到 {len(driver_files)} 个驱动图像")
        
        # 创建帧输出目录
        frames_dir = output_path / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # 创建中间结果输出目录
        source_recon_dir = output_path / "source_recon"
        source_recon_dir.mkdir(exist_ok=True)
        
        identity_driven_dir = output_path / "identity_driven"
        identity_driven_dir.mkdir(exist_ok=True)
        
        # 处理每个驱动图像
        edited_frames = []
        source_recon_frames = []
        identity_driven_frames = []
        
        for idx, driver_file in enumerate(tqdm(driver_files, desc="处理驱动图像")):
            # 加载驱动图像
            driver_img = Image.open(driver_file).convert("RGB")
            driver_tensor = transform_dict["test"](driver_img).unsqueeze(0).to(self.device)
            
            # 执行编辑
            results = self.inference_runner._run_editing_on_batch(
                source_results, driver_tensor, mask=mask
            )
            
            # 保存编辑后的图像和中间结果
            if results and "edited_images" in results and len(results["edited_images"]) > 0:
                # 1. 最终编辑结果
                edited_img_tensor = results["edited_images"][0][0]  # 获取第一个结果
                edited_img = tensor2im(edited_img_tensor)
                frame_path = frames_dir / f"frame_{idx:04d}.jpg"
                edited_img.save(frame_path)
                edited_frames.append(np.array(edited_img))
                
                # 2. 源图像重建 (x_E)
                source_recon_tensor = results["source_recon"][0]  # 获取第一个结果
                source_recon_img = tensor2im(source_recon_tensor)
                source_recon_path = source_recon_dir / f"source_recon_{idx:04d}.jpg"
                source_recon_img.save(source_recon_path)
                source_recon_frames.append(np.array(source_recon_img))
                
                # 3. 身份驱动 (y_E)
                identity_driven_tensor = results["identity_driven"][0]  # 获取第一个结果
                identity_driven_img = tensor2im(identity_driven_tensor)
                identity_driven_path = identity_driven_dir / f"identity_driven_{idx:04d}.jpg"
                identity_driven_img.save(identity_driven_path)
                identity_driven_frames.append(np.array(identity_driven_img))
            else:
                print(f"警告：处理驱动图像 {driver_file.name} 失败")
                if idx > 0 and edited_frames:
                    # 如果失败，使用前一帧
                    edited_frames.append(edited_frames[-1])
                    source_recon_frames.append(source_recon_frames[-1])
                    identity_driven_frames.append(identity_driven_frames[-1])
        
        # 如果源图像是对齐的，需要将结果还原
        if align_source:
            from runners.simple_runner import unalign
            unaligned_frames_dir = output_path / "unaligned_frames"
            unaligned_frames_dir.mkdir(exist_ok=True)
            
            # 对每一帧进行还原
            unaligned_frames = []
            for idx, edited_img in enumerate(tqdm(edited_frames, desc="还原帧")):
                pil_img = Image.fromarray(edited_img)
                unaligned_path = unaligned_frames_dir / f"unaligned_{idx:04d}.jpg"
                unalign(pil_img, unalign_dict, str(source_img_path), unaligned_path)
                unaligned_frames.append(np.array(Image.open(unaligned_path)))
            
            # 使用还原后的帧
            edited_frames = unaligned_frames
            frames_dir = unaligned_frames_dir
        
        # 如果没有生成任何帧，则退出
        if not edited_frames:
            print("错误：未能生成任何编辑帧")
            return
        
        # 创建视频文件名
        source_name = source_img_path.stem
        driver_name = driver_path.name
        
        # 1. 创建最终编辑结果视频
        final_video_path = output_path / f"{source_name}_driven_by_{driver_name}.mp4"
        self._create_video(edited_frames, final_video_path, fps)
        
        # 2. 创建源图像重建视频
        source_recon_video_path = output_path / f"{source_name}_source_recon.mp4"
        self._create_video(source_recon_frames, source_recon_video_path, fps)
        
        # 3. 创建身份驱动视频
        identity_driven_video_path = output_path / f"{source_name}_identity_driven.mp4"
        self._create_video(identity_driven_frames, identity_driven_video_path, fps)
        
        print(f"\n处理完成！")
        print(f"1. 最终编辑视频：{final_video_path}")
        print(f"2. 源图像重建视频：{source_recon_video_path}")
        print(f"3. 身份驱动视频：{identity_driven_video_path}")
        print(f"4. 驱动帧视频：{driver_video_path}")
        print(f"单帧图像已保存到对应目录")
        
        return final_video_path, frames_dir
        
    def _create_video(self, frames, output_path, fps):
        """
        从帧列表创建视频
        
        Args:
            frames: 帧列表
            output_path: 输出路径
            fps: 帧率
        """
        if not frames:
            print(f"警告：没有帧可用于创建视频 {output_path}")
            return
            
        print(f"创建视频: {output_path}")
        try:
            # 使用imageio创建视频
            writer = imageio.get_writer(
                str(output_path),
                format='FFMPEG',
                mode='I',
                fps=fps,
                codec='libx264',
                bitrate='5000k',
                quality=None,
                macro_block_size=None,
                output_params=['-pix_fmt', 'yuv420p']
            )
            
            for frame in frames:
                writer.append_data(frame)
            
            writer.close()
            print(f"视频创建成功：{output_path}")
            
        except Exception as e:
            print(f"使用imageio创建视频失败: {str(e)}")
            print("尝试使用OpenCV创建视频...")
            
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            video.release()
            print(f"使用OpenCV创建视频成功：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="驱动视频推理工具")
    parser.add_argument("--source", type=str, required=True, help="源图像路径")
    parser.add_argument("--driver_dir", type=str, required=True, help="驱动图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--editor_ckpt", type=str, required=True, help="编辑器模型检查点路径")
    parser.add_argument("--vivface_ckpt", type=str, required=True, help="ViVFace模型检查点路径")
    parser.add_argument("--inverter_ckpt", type=str, help="反转器模型检查点路径（可选，如果提供则优先使用）")
    parser.add_argument("--config", type=str, default="configs/driver_video_inference.yaml", help="配置文件路径")
    parser.add_argument("--fps", type=int, default=25, help="输出视频的帧率")
    parser.add_argument("--align", action="store_true", help="是否对源图像进行对齐")
    parser.add_argument("--use_mask", action="store_true", help="是否使用遮罩")
    parser.add_argument("--mask_threshold", type=float, default=0.995, help="遮罩阈值")
    parser.add_argument("--mask_path", type=str, default=None, help="自定义遮罩路径")
    
    args = parser.parse_args()
    
    # 初始化运行器
    runner = DriverVideoInferenceRunner(
        editor_ckpt_pth=args.editor_ckpt,
        vivface_ckpt_pth=args.vivface_ckpt,
        inverter_ckpt_pth=args.inverter_ckpt,
        config_pth=args.config,
        source_img_path=args.source,
        driver_dir=args.driver_dir,
        output_dir=args.output_dir
    )
    
    # 处理视频
    runner.process_video(
        source_img_path=args.source,
        driver_dir=args.driver_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        align_source=args.align,
        use_mask=args.use_mask,
        mask_threshold=args.mask_threshold,
        mask_path=args.mask_path
    )


if __name__ == "__main__":
    main()
