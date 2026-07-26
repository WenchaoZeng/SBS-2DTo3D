"""Real-ESRGAN 超分复原封装，用于去除图像/视频中的打码与像素化遮挡。

权重来自 Real-ESRGAN_x4plus，采用纯合成退化数据训练，对马赛克、模糊、
压缩噪声等退化具备较强的实用还原能力。
"""

import argparse
import sys
import types
from pathlib import Path

# 新版 torchvision（0.17+）移除了 functional_tensor 子模块，而 basicsr 1.4.2
# 仍从中导入 rgb_to_grayscale。此处注入兼容垫片，必须在导入 basicsr/realesrgan 之前执行。
import torchvision.transforms.functional as _vision_functional

if "torchvision.transforms.functional_tensor" not in sys.modules:
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _vision_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from interpolation.rife_model import select_device

MODEL_PATH = Path("models") / "realesrgan" / "RealESRGAN_x4plus.pth"


class RealESRGANRestorer:
    """加载 Real-ESRGAN 权重，提供对像素化/打码图像的超分复原接口。"""

    def __init__(self, model_path: str | Path = MODEL_PATH, scale: int = 4, outscale: int = 4, tile: int = 0) -> None:
        """读取权重并构建 RealESRGANer 推理器，按设备自动选择是否启用半精度。

        参数 scale 为模型固有的放大倍率（x4plus 对应 4）；outscale 为最终输出倍率，
        可小于 scale 以缩小结果；tile 为分块大小，0 表示不分块，大图可设 512 避免 MPS 显存溢出。
        """
        weights_path = Path(model_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"未找到 Real-ESRGAN 权重：{weights_path}")
        self.device = select_device()
        self.outscale = outscale
        # 与 RealESRGAN_x4plus 权重匹配的 RRDBNet 结构
        network = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        # fp16 半精度仅在 CUDA 上启用，MPS/CPU 使用 fp32 保证 grid_sample 等算子稳定
        use_half = self.device.type == "cuda"
        self.upsampler = RealESRGANer(scale=scale, model_path=str(weights_path), model=network, tile=tile, tile_pad=10, pre_pad=0, half=use_half, device=self.device)

    def restore(self, frame: np.ndarray) -> np.ndarray:
        """对一张 BGR uint8 帧做超分复原，返回复原后的 BGR uint8 帧。"""
        # RealESRGANer.enhance 接收 BGR uint8，返回 (输出帧, 实际放大倍率)
        output, _ = self.upsampler.enhance(frame, outscale=self.outscale)
        return output


def restore_image(input_path: str | Path, output_path: str | Path | None = None, outscale: int = 4, tile: int = 0) -> str:
    """对单张图片执行去马赛克/像素化复原，返回输出文件的绝对路径。"""
    source = Path(input_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入图片：{source}")
    frame = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"无法解码图片，可能不是有效图像：{source}")
    # RealESRGANer 内部按 RGB 三通道处理，带透明通道的图先剥离 Alpha 通道
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    destination = (Path(output_path) if output_path else Path("output") / "realesrgan" / f"{source.stem}_restored.png").expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    restorer = RealESRGANRestorer(outscale=outscale, tile=tile)
    result = restorer.restore(frame)
    cv2.imwrite(str(destination), result)
    return str(destination.resolve())


def main() -> None:
    """提供命令行入口：对单张图片做 Real-ESRGAN 去马赛克/像素化复原。"""
    parser = argparse.ArgumentParser(description="使用 Real-ESRGAN 对打码/像素化图片进行超分复原。")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--output", help="输出图片路径，默认输出到 output/realesrgan")
    parser.add_argument("--outscale", type=int, default=4, help="输出放大倍率，默认 4")
    parser.add_argument("--tile", type=int, default=0, help="分块大小，0 表示不分块，大图可设 512 防止显存溢出")
    arguments = parser.parse_args()
    output = restore_image(arguments.input, arguments.output, arguments.outscale, arguments.tile)
    print(f"输出文件：{output}")


if __name__ == "__main__":
    main()
