#!/usr/bin/env python3
"""
扫描 output 目录下所有未完成音视频合并的 SBS 任务,自动补合并。

典型场景:generate_sbs_video 处理长视频时,视频编码完成后、mux 步骤执行前
进程被中断(前端超时/会话断开/手动关闭),导致子目录里有 sbs_video_no_audio.mp4
和 audio.aac,但根目录缺最终合并文件。本脚本一键兜底补救。

用法:
    python3 remux_pending.py                              # 扫描 output 下全部 sbs_video_* 目录
    python3 remux_pending.py output/sbs_video_20260725_195212  # 只处理指定目录
"""
import os
import sys
import shutil
import subprocess
import glob

OUTPUT_DIR = "output"


def has_audio_stream(video_path):
    """检测视频文件是否包含音频流(用 ffprobe 查询音频流)"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=False
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except FileNotFoundError:
        print("未找到 ffprobe,请确保 ffmpeg 已安装并在 PATH 中")
        return False


def remux_one(work_dir):
    """
    补合并单个任务目录。
    work_dir 形如 output/sbs_video_20260725_195212
    返回本次处理结果的中文字符串描述
    """
    dirname = os.path.basename(os.path.normpath(work_dir))
    final_path = os.path.join(OUTPUT_DIR, f"{dirname}.mp4")
    no_audio_path = os.path.join(work_dir, "sbs_video_no_audio.mp4")
    audio_path = os.path.join(work_dir, "audio.aac")

    # 缺少编码产物,无法补救
    if not os.path.exists(no_audio_path):
        return f"跳过 {dirname}:缺少 sbs_video_no_audio.mp4"

    # 最终文件已存在且含音频流,视为已完成
    if os.path.exists(final_path) and os.path.getsize(final_path) > 0 and has_audio_stream(final_path):
        return f"跳过 {dirname}:最终文件已含音频"

    # 有音频则合并,无音频则直接落盘为最终文件
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        cmd = ['ffmpeg', '-y', '-i', no_audio_path, '-i', audio_path,
               '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
               '-shortest', final_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return f"失败 {dirname}:ffmpeg 合并出错\n{result.stderr[-500:]}"
        return f"成功 {dirname}:已合并音视频 -> {final_path}"

    shutil.copy2(no_audio_path, final_path)
    return f"成功 {dirname}:无音频,已落盘无音版 -> {final_path}"


def main():
    """入口:扫描所有待合并的 sbs_video_* 目录并逐个处理"""
    if not os.path.isdir(OUTPUT_DIR):
        print(f"输出目录不存在: {OUTPUT_DIR}")
        return

    # 传了参数则只处理指定目录;否则扫描 output 下全部 sbs_video_* 目录
    if len(sys.argv) > 1:
        dirs = [sys.argv[1]]
    else:
        dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "sbs_video_*")))

    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("没有找到需要处理的 sbs_video_* 目录")
        return

    print(f"共发现 {len(dirs)} 个任务目录,开始检查...\n")
    for d in dirs:
        print(remux_one(d))
    print("\n处理完成。")


if __name__ == "__main__":
    main()
