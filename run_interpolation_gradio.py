"""视频去噪与插帧的独立 Gradio 网页入口。"""

from pathlib import Path

import gradio as gr

from interpolation.process_video import process_video, probe_video


def read_video_info(video_path: str) -> str:
    """在用户填写路径后显示输入视频的基础信息。"""
    if not video_path or not Path(video_path).expanduser().is_file():
        return "请输入有效的视频文件路径。"
    try:
        fps, duration = probe_video(video_path)
    except Exception as error:
        return f"无法读取视频信息：{error}"
    return f"原始帧率：{fps:.3f} FPS；时长：{duration:.2f} 秒"


def process_interpolation(video_path: str, multiplier: int, enable_interpolation: bool, enable_denoise: bool, denoise_strength: float, progress=gr.Progress()) -> str:
    """将 Gradio 进度回调转交给独立的视频增强处理器。"""
    if not video_path:
        raise gr.Error("请填写输入视频文件路径。")

    def update_progress(percentage: float, message: str) -> None:
        """将 FFmpeg 进度展示在网页中。"""
        progress(percentage, desc=message)

    try:
        return process_video(video_path, multiplier, enable_interpolation, enable_denoise, denoise_strength, progress_callback=update_progress)
    except Exception as error:
        raise gr.Error(str(error)) from error


def create_demo() -> gr.Blocks:
    """创建包含去噪与插帧勾选项的独立网页界面。"""
    with gr.Blocks(title="视频增强") as demo:
        gr.Markdown("## 视频增强：去噪与插帧")
        gr.Markdown("按需使用 FastDVDnet 时域去噪与 RIFE 4.25 AI 插帧。两项都勾选时先去噪再插帧；输出为 H.264/AAC MP4，保留原音频。")
        with gr.Row():
            with gr.Column():
                video_path = gr.Textbox(
                    label="输入视频文件路径",
                    placeholder="例如：input/video.mp4",
                    lines=1,
                )
                video_info = gr.Textbox(label="视频信息", interactive=False)
                multiplier = gr.Dropdown(
                    choices=[2, 3, 4],
                    value=2,
                    label="插帧倍率",
                    info="输出帧率等于原始帧率乘以该倍率，例如 30 FPS 的 2 倍为 60 FPS。",
                )
                enable_interpolation = gr.Checkbox(label="启用 RIFE AI 插帧", value=True)
                enable_denoise = gr.Checkbox(label="启用 FastDVDnet 视频去噪", value=False)
                denoise_strength = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="去噪强度", info="噪点较多时提高；数值过高可能抹除细节。")
                process_button = gr.Button("开始视频处理", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="处理输出", interactive=False)

        video_path.change(read_video_info, inputs=video_path, outputs=video_info)
        process_button.click(
            process_interpolation,
            inputs=[video_path, multiplier, enable_interpolation, enable_denoise, denoise_strength],
            outputs=output_video,
        )
    return demo


if __name__ == "__main__":
    # 使用独立端口，避免与现有 SBS 页面默认的 7860 端口冲突。
    create_demo().queue(default_concurrency_limit=1).launch(server_port=7861)
