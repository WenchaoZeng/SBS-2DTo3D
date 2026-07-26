"""RIFE 4.25 推理网络封装，模型权重来自 Practical-RIFE。"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


def select_device() -> torch.device:
    """优先选择 CUDA，其次使用 Apple Silicon 的 MPS，最后回退 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """按光流场反向采样图像，网格始终创建在输入张量所在设备。"""
    height, width = flow.shape[2:]
    horizontal = torch.linspace(-1.0, 1.0, width, device=image.device).view(1, 1, 1, width)
    vertical = torch.linspace(-1.0, 1.0, height, device=image.device).view(1, 1, height, 1)
    grid = torch.cat((horizontal.expand(flow.shape[0], -1, height, -1), vertical.expand(flow.shape[0], -1, -1, width)), 1)
    normalized_flow = torch.cat((flow[:, :1] / ((image.shape[3] - 1.0) / 2.0), flow[:, 1:] / ((image.shape[2] - 1.0) / 2.0)), 1)
    return functional.grid_sample(image, (grid + normalized_flow).permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """构建 RIFE 使用的卷积与 LeakyReLU 单元。"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=True), nn.LeakyReLU(0.2, True))


class Head(nn.Module):
    """提取两个输入帧的浅层特征。"""

    def __init__(self) -> None:
        """初始化编码头。"""
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """返回用于光流估计的四通道特征。"""
        value = self.relu(self.cnn0(value))
        value = self.relu(self.cnn1(value))
        value = self.relu(self.cnn2(value))
        return self.cnn3(value)


class ResConv(nn.Module):
    """RIFE 光流块中的残差卷积单元。"""

    def __init__(self, channels: int) -> None:
        """初始化残差卷积及其可学习缩放参数。"""
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """执行残差特征更新。"""
        return self.relu(self.conv(value) * self.beta + value)


class IFBlock(nn.Module):
    """在指定尺度上估计并细化双向光流。"""

    def __init__(self, in_planes: int, channels: int) -> None:
        """初始化 RIFE 的多尺度光流块。"""
        super().__init__()
        self.conv0 = nn.Sequential(conv(in_planes, channels // 2, 3, 2, 1), conv(channels // 2, channels, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(channels) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(channels, 52, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, value: torch.Tensor, flow: torch.Tensor | None, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回当前尺度估计的光流、遮罩与中间特征。"""
        value = functional.interpolate(value, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = functional.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            value = torch.cat((value, flow), 1)
        feature = self.convblock(self.conv0(value))
        result = functional.interpolate(self.lastconv(feature), scale_factor=scale, mode="bilinear", align_corners=False)
        return result[:, :4] * scale, result[:, 4:5], result[:, 5:]


class IFNet(nn.Module):
    """与 RIFE 4.25 权重对应的五级光流推理网络。"""

    def __init__(self) -> None:
        """初始化所有多尺度光流块。"""
        super().__init__()
        self.blocks = nn.ModuleList((IFBlock(15, 192), IFBlock(28, 128), IFBlock(28, 96), IFBlock(28, 64), IFBlock(28, 32)))
        self.encode = Head()

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, timestep: float, scale: float) -> torch.Tensor:
        """根据两个相邻帧和时间位置生成中间帧。"""
        timestep_tensor = torch.full_like(image0[:, :1], timestep)
        feature0, feature1 = self.encode(image0), self.encode(image1)
        scales = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow = mask = feature = None
        warped0, warped1 = image0, image1
        for index, block in enumerate(self.blocks):
            if flow is None:
                flow, mask, feature = block(torch.cat((image0, image1, feature0, feature1, timestep_tensor), 1), None, scales[index])
                warped0, warped1 = warp(image0, flow[:, :2]), warp(image1, flow[:, 2:])
                continue
            warped_feature0, warped_feature1 = warp(feature0, flow[:, :2]), warp(feature1, flow[:, 2:])
            delta_flow, mask, feature = block(torch.cat((warped0, warped1, warped_feature0, warped_feature1, timestep_tensor, mask, feature), 1), flow, scales[index])
            flow = flow + delta_flow
            warped0, warped1 = warp(image0, flow[:, :2]), warp(image1, flow[:, 2:])
        return warped0 * torch.sigmoid(mask) + warped1 * (1 - torch.sigmoid(mask))


class RifeInterpolator:
    """加载本地 RIFE 4.25 权重并提供 OpenCV 帧插帧接口。"""

    def __init__(self, model_path: str | Path, scale: float = 1.0) -> None:
        """读取模型权重并将网络移动到可用的推理设备。"""
        weights_path = Path(model_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"未找到 RIFE 权重：{weights_path}")
        self.device = select_device()
        self.scale = scale
        self.network = IFNet().to(self.device).eval()
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state_dict, strict=False)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, timestep: float) -> np.ndarray:
        """为两张 BGR OpenCV 帧生成指定时间位置的 BGR 中间帧。"""
        height, width = frame0.shape[:2]
        padded_height = ((height - 1) // 128 + 1) * 128
        padded_width = ((width - 1) // 128 + 1) * 128
        padding = (0, padded_width - width, 0, padded_height - height)
        image0 = torch.from_numpy(np.ascontiguousarray(frame0[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0).to(self.device).float() / 255.0
        image1 = torch.from_numpy(np.ascontiguousarray(frame1[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0).to(self.device).float() / 255.0
        with torch.inference_mode():
            result = self.network(functional.pad(image0, padding), functional.pad(image1, padding), timestep, self.scale)
        rgb = (result[0, :, :height, :width].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
