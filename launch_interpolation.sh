#!/usr/bin/env bash

# 启动独立的视频插帧网页，默认访问地址为 http://127.0.0.1:7861 。
source venv/bin/activate
python3 run_interpolation_gradio.py
