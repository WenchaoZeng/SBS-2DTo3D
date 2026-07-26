"""使用 RIFE AI 模型执行视频插帧的独立处理脚本。"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import cv2

try:
    # 作为项目模块导入时使用绝对路径，供 Gradio 页面调用。
    from interpolation.rife_model import RifeInterpolator
except ModuleNotFoundError:
    # 直接执行 python interpolation/process_video.py 时，脚本目录位于导入路径首位。
    from rife_model import RifeInterpolator
    from fastdvdnet_model import FastDVDnetDenoiser
else:
    from interpolation.fastdvdnet_model import FastDVDnetDenoiser


ProgressCallback = Callable[[float, str], None]
MODEL_PATH = Path("models") / "rife" / "train_log" / "flownet.pkl"
FASTDVDNET_MODEL_PATH = Path("models") / "fastdvdnet" / "model.pth"


def require_ffmpeg() -> tuple[str, str]:
    """检查 FFmpeg 与 FFprobe 是否可用，并返回它们的可执行路径。"""
    ffmpeg_path, ffprobe_path = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError("未找到 ffmpeg 或 ffprobe，请先安装 FFmpeg 并确保其位于 PATH 中。")
    return ffmpeg_path, ffprobe_path


def probe_video(video_path: str | Path) -> tuple[float, float, int]:
    """读取视频帧率、时长和总帧数，用于输出设置及进度计算。"""
    _, ffprobe_path = require_ffmpeg()
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,nb_frames:format=duration", "-of", "json", str(video_path)], capture_output=True, text=True, check=True)
    metadata, streams = json.loads(result.stdout), json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("输入文件不包含视频流。")
    numerator, denominator = streams[0]["avg_frame_rate"].split("/")
    fps, duration = float(numerator) / float(denominator), float(metadata["format"]["duration"])
    frame_count = int(streams[0].get("nb_frames") or round(fps * duration))
    if fps <= 0 or duration <= 0 or frame_count < 2:
        raise ValueError("无法读取输入视频的有效帧率、时长或帧数。")
    return fps, duration, frame_count


def default_output_path(input_path: str | Path, multiplier: int) -> Path:
    """生成位于 output/interpolation 下且包含插帧倍率的输出路径。"""
    source = Path(input_path)
    output_dir = Path("output") / "interpolation"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{source.stem}_{multiplier}x_rife.mp4"


def merge_audio(video_path: Path, source_path: Path, output_path: Path) -> None:
    """将原视频音轨与 RIFE 生成的视频流合并为兼容性良好的 MP4。"""
    ffmpeg_path, _ = require_ffmpeg()
    command = [ffmpeg_path, "-y", "-i", str(video_path), "-i", str(source_path), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-map_metadata", "1", "-movflags", "+faststart", str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("音视频合并失败：" + "\n".join(result.stderr.splitlines()[-8:]))


def denoised_frame_stream(capture: cv2.VideoCapture, denoiser: FastDVDnetDenoiser):
    """以五帧滑动窗口流式产出 FastDVDnet 去噪帧，避免缓存完整视频。"""
    initial_frames = []
    for _ in range(3):
        success, frame = capture.read()
        if not success:
            raise RuntimeError("视频帧数量不足，无法执行 FastDVDnet 去噪。")
        initial_frames.append(frame)
    yield denoiser.denoise([initial_frames[0], initial_frames[0], *initial_frames])
    window = initial_frames
    while True:
        success, frame = capture.read()
        if not success:
            break
        window.append(frame)
        if len(window) == 4:
            yield denoiser.denoise([window[0], *window])
            continue
        yield denoiser.denoise(window)
        window = window[1:]
    yield denoiser.denoise([window[0], window[1], window[2], window[2], window[2]])
    yield denoiser.denoise([window[1], window[2], window[2], window[2], window[2]])


def process_video(input_path: str | Path, multiplier: int, enable_interpolation: bool, enable_denoise: bool, noise_sigma: float = 20, output_path: str | Path | None = None, scale: float = 1.0, progress_callback: ProgressCallback | None = None) -> str:
    """按勾选项执行 FastDVDnet 去噪、RIFE 插帧或二者串联，并保留原始音频。"""
    source = Path(input_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入视频：{source}")
    if multiplier not in (2, 3, 4):
        raise ValueError("插帧倍率仅支持 2、3 或 4 倍。")
    if not enable_interpolation and not enable_denoise:
        raise ValueError("请至少勾选视频插帧或视频去噪其中一项。")
    if scale not in (0.5, 1.0):
        raise ValueError("RIFE 推理缩放仅支持 0.5 或 1.0。")

    source_fps, _, frame_count = probe_video(source)
    suffix = f"{multiplier}x_rife" if enable_interpolation else "fastdvdnet_denoised"
    destination = (Path(output_path) if output_path else Path("output") / "interpolation" / f"{source.stem}_{suffix}.mp4").expanduser().with_suffix(".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(0, "正在加载所选 AI 模型到推理设备。")
    denoiser = FastDVDnetDenoiser(FASTDVDNET_MODEL_PATH, noise_sigma) if enable_denoise else None
    interpolator = RifeInterpolator(MODEL_PATH, scale) if enable_interpolation else None
    capture = cv2.VideoCapture(str(source))
    if enable_denoise:
        frames = denoised_frame_stream(capture, denoiser)
        try:
            previous_frame = next(frames)
        except StopIteration as error:
            capture.release()
            raise RuntimeError("无法解码输入视频的首帧。") from error
    else:
        def raw_frame_stream():
            """逐帧读取原视频，供仅插帧的路径使用。"""
            while True:
                success, frame = capture.read()
                if not success:
                    return
                yield frame
        frames = raw_frame_stream()
        try:
            previous_frame = next(frames)
        except StopIteration as error:
            capture.release()
            raise RuntimeError("无法解码输入视频的首帧。") from error
    height, width = previous_frame.shape[:2]
    output_fps = source_fps * multiplier if enable_interpolation else source_fps
    # 临时无声视频写入 output 下的临时目录，便于运行中查看进度和异常恢复
    temp_directory = Path("output") / "interpolation" / f"{destination.stem}_tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    silent_video = temp_directory / "video.mp4"
    writer = cv2.VideoWriter(str(silent_video), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建临时视频编码器。")
    processed = 1
    try:
        for current_frame in frames:
            writer.write(previous_frame)
            if enable_interpolation:
                for index in range(1, multiplier):
                    writer.write(interpolator.interpolate(previous_frame, current_frame, index / multiplier))
            previous_frame = current_frame
            processed += 1
            if progress_callback:
                percentage = processed / frame_count * 0.9
                progress_callback(percentage, f"AI 视频处理中：{processed}/{frame_count} 帧")
        # OpenCV VideoWriter 不支持显式尾帧时长；重复末帧以保持输出时长与原视频一致。
        for _ in range(multiplier if enable_interpolation else 1):
            writer.write(previous_frame)
    finally:
        capture.release()
        writer.release()
    if progress_callback:
        progress_callback(0.92, "正在编码 H.264 视频并合并原始音频。")
    merge_audio(silent_video, source, destination)
    # 音视频合并成功后清理临时无声视频
    shutil.rmtree(temp_directory, ignore_errors=True)
    if progress_callback:
        progress_callback(1, f"处理完成：{source_fps:.3f} FPS -> {output_fps:.3f} FPS")
    return str(destination.resolve())


def main() -> None:
    """提供无需网页的命令行 RIFE 插帧入口。"""
    parser = argparse.ArgumentParser(description="使用 FastDVDnet 去噪与 RIFE 4.25 插帧处理视频。")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--multiplier", type=int, default=2, choices=(2, 3, 4), help="帧率倍率")
    parser.add_argument("--output", help="输出 MP4 路径，默认输出到 output/interpolation")
    parser.add_argument("--scale", type=float, default=1.0, choices=(0.5, 1.0), help="4K 视频可使用 0.5 降低显存占用")
    parser.add_argument("--denoise", action="store_true", help="启用 FastDVDnet 时域去噪")
    parser.add_argument("--denoise-strength", type=float, default=20, help="FastDVDnet 去噪强度，范围 1-50")
    parser.add_argument("--no-interpolation", action="store_true", help="仅去噪，不执行 RIFE 插帧")
    arguments = parser.parse_args()
    output = process_video(arguments.input, arguments.multiplier, not arguments.no_interpolation, arguments.denoise, arguments.denoise_strength, arguments.output, arguments.scale, lambda _, message: print(message, flush=True))
    print(f"输出文件：{output}")


if __name__ == "__main__":
    main()
