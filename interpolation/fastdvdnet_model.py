"""FastDVDnet 视频时域去噪模型封装。"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

try:
    from interpolation.rife_model import select_device
except ModuleNotFoundError:
    from rife_model import select_device


class CvBlock(nn.Module):
    """构建 FastDVDnet 编解码器使用的双卷积特征块。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """初始化与官方权重键名一致的特征块。"""
        super().__init__()
        self.convblock = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """提取当前尺度的图像特征。"""
        return self.convblock(value)


class InputCvBlock(nn.Module):
    """分别处理三帧图像与噪声图的输入特征块。"""

    def __init__(self, num_in_frames: int, out_ch: int) -> None:
        """初始化分组卷积输入层。"""
        super().__init__()
        intermediate_channels = 30
        self.convblock = nn.Sequential(nn.Conv2d(num_in_frames * 4, num_in_frames * intermediate_channels, 3, padding=1, groups=num_in_frames, bias=False), nn.BatchNorm2d(num_in_frames * intermediate_channels), nn.ReLU(True), nn.Conv2d(num_in_frames * intermediate_channels, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """提取输入图像的初始特征。"""
        return self.convblock(value)


class DownBlock(nn.Module):
    """执行下采样与双卷积特征提取。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """初始化下采样块。"""
        super().__init__()
        self.convblock = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True), CvBlock(out_ch, out_ch))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """压缩空间尺寸并提取特征。"""
        return self.convblock(value)


class UpBlock(nn.Module):
    """执行双卷积特征提取与 PixelShuffle 上采样。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """初始化上采样块。"""
        super().__init__()
        self.convblock = nn.Sequential(CvBlock(in_ch, in_ch), nn.Conv2d(in_ch, out_ch * 4, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """恢复更高分辨率特征。"""
        return self.convblock(value)


class OutputCvBlock(nn.Module):
    """将解码特征转换为三通道噪声残差。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """初始化输出卷积层。"""
        super().__init__()
        self.convblock = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(True), nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """预测需要从输入图像去除的噪声。"""
        return self.convblock(value)


class DenBlock(nn.Module):
    """FastDVDnet 的三帧时域去噪块。"""

    def __init__(self, num_input_frames: int = 3) -> None:
        """初始化与官方预训练权重完全一致的 U-Net 结构。"""
        super().__init__()
        self.inc = InputCvBlock(num_input_frames, 32)
        self.downc0 = DownBlock(32, 64)
        self.downc1 = DownBlock(64, 128)
        self.upc2 = UpBlock(128, 64)
        self.upc1 = UpBlock(64, 32)
        self.outc = OutputCvBlock(32, 3)

    def forward(self, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        """利用三帧和噪声图估计中间帧的干净结果。"""
        value0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), 1))
        value1 = self.downc0(value0)
        value2 = self.downc1(value1)
        value2 = self.upc2(value2)
        value1 = self.upc1(value1 + value2)
        return in1 - self.outc(value0 + value1)


class FastDVDnet(nn.Module):
    """由两级三帧去噪网络组成的 FastDVDnet 模型。"""

    def __init__(self, num_input_frames: int = 5) -> None:
        """初始化两阶段时域去噪网络。"""
        super().__init__()
        self.num_input_frames = num_input_frames
        self.temp1 = DenBlock(3)
        self.temp2 = DenBlock(3)

    def forward(self, value: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        """对连续五帧执行两阶段时域去噪并返回中心帧。"""
        frame0, frame1, frame2, frame3, frame4 = (value[:, 3 * index:3 * index + 3] for index in range(self.num_input_frames))
        stage0 = self.temp1(frame0, frame1, frame2, noise_map)
        stage1 = self.temp1(frame1, frame2, frame3, noise_map)
        stage2 = self.temp1(frame2, frame3, frame4, noise_map)
        return self.temp2(stage0, stage1, stage2, noise_map)


class FastDVDnetDenoiser:
    """加载本地 FastDVDnet 权重，并以五帧滑动窗口执行视频去噪。"""

    def __init__(self, model_path: str | Path, noise_sigma: float) -> None:
        """加载预训练权重，并校验用户选择的噪声强度。"""
        if not 0 < noise_sigma <= 50:
            raise ValueError("去噪强度必须在 1 到 50 之间。")
        weights_path = Path(model_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"未找到 FastDVDnet 权重：{weights_path}")
        self.device = select_device()
        self.noise_sigma = noise_sigma / 255.0
        self.network = FastDVDnet().to(self.device).eval()
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.network.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()}, strict=True)
        # fp16 半精度推理：MPS/CUDA 上可显著提升速度并降低显存占用
        self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        if self.dtype == torch.float16:
            self.network = self.network.half()

    def denoise(self, frames: list[np.ndarray]) -> np.ndarray:
        """对五张 BGR 帧去噪，返回其中心帧的 BGR 去噪结果。"""
        if len(frames) != 5:
            raise ValueError("FastDVDnet 去噪需要连续五帧。")
        height, width = frames[0].shape[:2]
        padded_height, padded_width = ((height + 3) // 4) * 4, ((width + 3) // 4) * 4
        tensors = [torch.from_numpy(np.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1))).to(self.device, dtype=self.dtype) / 255.0 for frame in frames]
        value = torch.cat(tensors).unsqueeze(0)
        padding = (0, padded_width - width, 0, padded_height - height)
        noise_map = torch.full((1, 1, height, width), self.noise_sigma, device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.network(functional.pad(value, padding, mode="reflect"), functional.pad(noise_map, padding, mode="reflect"))
        rgb = (output[0, :, :height, :width].clamp(0, 1).float() * 255).byte().cpu().numpy().transpose(1, 2, 0)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
