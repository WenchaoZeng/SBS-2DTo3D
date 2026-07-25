import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import time
from utils.colored_print import color, style
from safetensors.torch import load_file as load_safetensors # Import safetensors loading function
import matplotlib as mpl # Import matplotlib for colormap
# import matplotlib.pyplot as plt  # Import matplotlib for colormap (old)
# from matplotlib import cm  # Import colormap module
import cv2
import tempfile
import shutil
import imageio
import subprocess
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming the depth_anything_v2 directory is in the same folder as main.py
from depth_anything_v2.dpt import DepthAnythingV2
from sbs.sbs import process_image_sbs # Import for SBS processing

# Model configurations (similar to the original node)
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Available models and their Hugging Face filenames
AVAILABLE_MODELS = [
    'depth_anything_v2_vits_fp16.safetensors',
    'depth_anything_v2_vits_fp32.safetensors',
    'depth_anything_v2_vitb_fp16.safetensors',
    'depth_anything_v2_vitb_fp32.safetensors',
    'depth_anything_v2_vitl_fp16.safetensors',
    'depth_anything_v2_vitl_fp32.safetensors'
]

# AVAILABLE_MODELS = [
#     'depth_anything_v2_vitl_fp16.safetensors',    
#     'depth_anything_v2_vitb_fp16.safetensors',
#     'depth_anything_v2_vits_fp16.safetensors'        
# ]


# load depth-anything-2 model
def load_model(model_name, device, models_dir='models/depthanything'):
    """Loads the specified Depth Anything V2 model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {AVAILABLE_MODELS}")

    print(f"Selected model: {model_name}")
    dtype = torch.float16 if "fp16" in model_name else torch.float32
    encoder = 'vitl' # Default
    if "vitl" in model_name:
        encoder = "vitl"
    elif "vitb" in model_name:
        encoder = "vitb"
    elif "vits" in model_name:
        encoder = "vits"

    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name} to {model_path}...")
        os.makedirs(models_dir, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="yushan777/DepthAnythingV2",
                filename=model_name,
                local_dir=models_dir,
                local_dir_use_symlinks=False
            )
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    print(f"Loading model from: {model_path}")
    # Use safetensors.torch.load_file for .safetensors files
    state_dict = load_safetensors(model_path, device='cpu')

    max_depth = 20.0 if "hypersim" in model_name else 80.0
    is_metric = 'metric' in model_name

    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**{**config, 'is_metric': is_metric, 'max_depth': max_depth})

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device=device, dtype=dtype)
    print("Model loaded successfully.")

    return model, dtype, is_metric

def process_depthmap_image(model, image_tensor, device, dtype, is_metric, output_filename_base, output_dir_frames="output", save_to_disk=True): # Added output_dir_frames with default for backward compatibility if called elsewhere
    # performs the core inference, and post-processes the raw depth output by (normalization, resizing), 
    # converts it to a PIL image, and saves it.


    # Ensure dimensions are divisible by 14
    # if not, then they will be resize to the nearest multiple of 14.  
    # the final depth-map image will be resized to match dims of the original input image
    orig_H, orig_W = image_tensor.shape[2:]
    new_H, new_W = orig_H, orig_W
    if new_W % 14 != 0:
        new_W = new_W - (new_W % 14)
    if new_H % 14 != 0:
        new_H = new_H - (new_H % 14)

    if new_H != orig_H or new_W != orig_W:
        print(f"Resizing input from {orig_W}x{orig_H} to {new_W}x{new_H} to be multiple of 14")
        image_tensor = F.interpolate(image_tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)

    # Inference
    start_time = time.time()
    if device.type == 'cuda': # Reset peak memory stats before inference if using CUDA
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        depth = model(image_tensor)
        
    end_time = time.time()
    print(f"Inference took {end_time - start_time:.2f} seconds")

    if device.type == 'cuda': # Report peak CUDA memory usage after inference
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mib = peak_memory_bytes / (1024 * 1024) 
        peak_memory_gib = peak_memory_bytes / (1024 * 1024 * 1024)
        # print(f"Peak GPU Memory Allocated during inference: {peak_memory_mib:.2f} MiB ({peak_memory_gib:.2f} GiB)")

    # Postprocessing
    depth = depth.squeeze(0).squeeze(0) # Remove batch (dim 0) and channel (dim 0 again after first squeeze) -> (H, W)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) # Normalize to 0-1 -> (H, W)

    # Resize back to original (or slightly adjusted) size
    # Ensure final dimensions are even for potential later use
    final_H = (orig_H // 2) * 2
    final_W = (orig_W // 2) * 2
    # Check shape using correct indices for 2D tensor
    if depth.ndim == 2 and (depth.shape[0] != final_H or depth.shape[1] != final_W):
         # Interpolate: expects NCHW, add N and C dims. Squeeze back to HW.
         depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(final_H, final_W), mode="bilinear", align_corners=False).squeeze()

    depth = torch.clamp(depth, 0, 1)

    if is_metric:
        depth = 1.0 - depth # Invert for metric models

    # Convert to numpy array and scale to 0-255
    depth_np = depth.cpu().numpy()
    depth_visual = (depth_np * 255).astype(np.uint8)

    # Create PIL image (grayscale)
    depth_image = Image.fromarray(depth_visual)

    # # COLOR DEPTH MAP
    # # Apply colormap (Spectral_r to match original implementation)
    # cmap = mpl.colormaps['Spectral_r']
    # colored_depth = cmap(depth_np)[:, :, :3]  # Remove alpha channel
    # colored_depth = (colored_depth * 255).astype(np.uint8)
    # # Convert to PIL image
    # colored_depth_image = Image.fromarray(colored_depth)

    # save the image(s) into the output directory before returning
    # output_dir = "output" # Old hardcoded output
    # 仅在需要落盘时才创建目录并保存（视频流式处理时 save_to_disk=False，避免每帧 I/O）
    if save_to_disk:
        os.makedirs(output_dir_frames, exist_ok=True) # Use new parameter

        # Save grayscale depth map
        grayscale_path = os.path.join(output_dir_frames, f"{output_filename_base}_depth.png")
        depth_image.save(grayscale_path)
        print(f"Saved grayscale depth map to: {grayscale_path}")

        # Save colored depth map
        colored_path = os.path.join(output_dir_frames, f"{output_filename_base}_depth_colored.png") # Use new parameter
        # colored_depth_image.save(colored_path)
        # print(f"Saved colored depth map to: {colored_path}")

    return depth_image

def generate_depth_map_only(input_image, model_name):
    """
    Generates only the depth map from the input image.
    called by generate_depth_and_sbs_combined()
    """
    if input_image is None:
        gr.Warning("Please upload an image for depth map generation.")
        return None
    if model_name is None:
        gr.Warning("Please select a model for depth map generation.")
        return None

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU detected for depth map. Using GPU.")
    elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Apple Silicon GPU detected for depth map. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU detected for depth map. Using CPU.")

    # Load model
    try:
        model, dtype, is_metric = load_model(model_name, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        gr.Error(f"Failed to load model: {e}")
        return None

    # Create a working copy of the input image for depth processing
    image_for_depth_processing = input_image.copy()

    # Preprocessing
    transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        image_tensor = transform_normalize(image_for_depth_processing).unsqueeze(0).to(device=device, dtype=dtype)
    except Exception as e:
        print(f"Error during image transformation: {e}")
        gr.Error(f"Error during image transformation: {e}")
        return None

    # Generate output filename base
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = f"gradio_image_{timestamp}"
    
    # Process image for depth map
    try:
        depth_image_pil = process_depthmap_image(model, image_tensor, device, dtype, is_metric, output_filename_base)
        if depth_image_pil:
            print("Depth map generated successfully.")
        return depth_image_pil
    except Exception as e:
        print(f"Error processing image for depth map: {e}")
        gr.Error(f"Error processing image for depth map: {e}")
        return None

def generate_sbs_image_from_depth(original_input_image, depth_map_pil, model_name, sbs_method, sbs_depth_scale, sbs_mode, sbs_depth_blur_strength):
    """Generates the SBS 3D image using the original image and a pre-generated depth map."""
    if original_input_image is None:
        gr.Warning("Please provide the original input image for SBS generation.")
        return None
    if depth_map_pil is None:
        gr.Warning("Please generate or provide a depth map for SBS generation.")
        return None
    if model_name is None: # Needed for dtype
        gr.Warning("Please select a model (needed for data type).")
        return None


    # Determine device (can be different from depth map generation if run separately)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU detected for SBS. Using GPU.")
    elif torch.backends.mps.is_available():
         device = torch.device("mps")
         print("Apple Silicon GPU detected for SBS. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU detected for SBS. Using CPU.")

    # Determine dtype from model_name (as load_model is not called here directly for the primary model)
    dtype = torch.float16 if "fp16" in model_name else torch.float32

    try:
        # Prepare base_image for SBS: PIL to Tensor [1, H, W, C], float16, range [0,1]
        if original_input_image.mode != 'RGB':
            original_input_image = original_input_image.convert('RGB')
        
        transform_to_tensor = transforms.ToTensor() # Converts PIL [0,255] to Tensor [0,1]
        base_image_for_sbs = transform_to_tensor(original_input_image).permute(1, 2, 0).unsqueeze(0)
        # Ensure correct dtype for SBS processing, which expects float16
        base_image_for_sbs = base_image_for_sbs.to(device=device, dtype=torch.float16)


        # Prepare depth_map for SBS: PIL to Tensor [1, H, W, 1], float16, range [0,1]
        # depth_map_pil is already grayscale
        depth_map_for_sbs = transform_to_tensor(depth_map_pil).permute(1, 2, 0).unsqueeze(0)
        # Ensure correct dtype for SBS processing
        depth_map_for_sbs = depth_map_for_sbs.to(device=device, dtype=torch.float16)
        
        # Ensure depth_blur_strength is odd for SBS
        if sbs_depth_blur_strength % 2 == 0:
            sbs_depth_blur_strength +=1
            gr.Info(f"SBS Depth Blur Strength adjusted to {sbs_depth_blur_strength} (must be odd-numbered).")

        # print(f"Calling process_image_sbs with method: {sbs_method}, scale: {sbs_depth_scale}, mode: {sbs_mode}, blur: {sbs_depth_blur_strength}")
        sbs_image_tensor = process_image_sbs(
                base_image=base_image_for_sbs,
                depth_map=depth_map_for_sbs,
                method=sbs_method,
                depth_scale=sbs_depth_scale,
                mode=sbs_mode,
                depth_blur_strength=sbs_depth_blur_strength
            )
        
        # print(f"[run_gradio.generate_sbs_from_depth] sbs_image_tensor shape: {sbs_image_tensor.shape}", color.YELLOW)
        sbs_image_pil = transforms.ToPILImage()(sbs_image_tensor.squeeze(0).cpu().permute(2, 0, 1))
        print("SBS image generated successfully.")
        return sbs_image_pil
    
    except Exception as e:
        print(f"Error generating SBS image: {e}")
        gr.Error(f"Error generating SBS image: {e}")
        return None


def generate_depth_and_sbs_combined(input_image, model_name, sbs_method, sbs_depth_scale, sbs_mode, sbs_depth_blur_strength):
    """Combined function that generates depth map and then SBS image in sequence."""
    
    # Step 1: Generate depth map
    print("Step 1: Generating depth map...")
    depth_map = generate_depth_map_only(input_image, model_name)
    
    if depth_map is None:
        return None, None  # Return None for both outputs if depth map generation fails
    
    # makesure 
    # Step 2: Generate SBS image using the generated depth map
    print("Step 2: Generating SBS 3D image...")
    sbs_image = generate_sbs_image_from_depth(
        input_image, 
        depth_map, 
        model_name, 
        sbs_method, 
        sbs_depth_scale, 
        sbs_mode, 
        sbs_depth_blur_strength
    )
    
    return depth_map, sbs_image  # Return both outputs

def convert_ts_to_mp4(video_path):
    """
    将 TS 格式视频转换为 MP4 格式，以便 OpenCV 能正常读取。
    使用 VideoToolbox 硬件加速。
    转换后的文件输出到 output 目录。
    返回转换后的 MP4 文件路径；如果转换失败则返回原始路径。
    """
    # 检查文件扩展名是否为 TS 格式
    if not video_path.lower().endswith('.ts'):
        return video_path

    print(f"检测到 TS 格式视频，正在转换为 MP4 格式...")
    # 确保 output 目录存在
    os.makedirs("output", exist_ok=True)
    # 生成转换后的输出文件路径（输出到 output 目录）
    filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp4"
    mp4_path = os.path.join("output", filename)

    # 如果目标 MP4 文件已存在（且非空），则直接复用，跳过重复转换
    if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
        print(f"已存在转换后的 MP4 文件，跳过转换：{mp4_path}")
        return mp4_path

    try:
        # macOS 使用 VideoToolbox 硬件加速
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-c:v', 'h264_videotoolbox',
            '-b:v', '8M',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            mp4_path
        ]
        print("使用 VideoToolbox 硬件加速编码...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"TS 视频已成功转换为: {mp4_path}")
        return mp4_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg 转换 TS 视频失败: {e.stderr}")
        return video_path
    except FileNotFoundError:
        print("未找到 FFmpeg，无法转换 TS 视频。请确保 FFmpeg 已安装并在 PATH 中。")
        return video_path


def generate_sbs_video(video_path, model_name, sbs_method, sbs_mode, sbs_depth_scale, sbs_depth_blur_strength, start_time=0, end_time=0, progress=gr.Progress(track_tqdm=True)):
    if not video_path:
        gr.Warning("Please upload a video to process.")
        return None

    # 如果是 TS 格式，先转换为 MP4
    video_path = convert_ts_to_mp4(video_path)

    # 1. Setup (device, dtype, load depth model)
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")
    print(f"Using device: {device} for video processing.")
    
    dtype = torch.float16 if "fp16" in model_name else torch.float32
    
    try:
        depth_model, _, is_metric = load_model(model_name, device) # Unpack model, dtype (ignore), is_metric
    except Exception as e:
        gr.Error(f"Failed to load model: {e}")
        return None

    # 2. Create Working Directories (所有中间文件保存在 output 目录下，便于查看)
    work_timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_video_base_name = f"sbs_video_{work_timestamp}.mp4"
    final_output_video_path = os.path.join("output", output_video_base_name) # Ensure "output" dir exists
    os.makedirs("output", exist_ok=True)
    # 中间文件目录与最终输出视频使用同一时间戳，便于对应查看
    work_parent_dir = os.path.join("output", f"sbs_video_{work_timestamp}")
    # 仅保留 SBS 帧目录（原始帧和深度图帧在内存中流式处理，不再落盘）
    frames_sbs_dir = os.path.join(work_parent_dir, "frames_sbs")
    os.makedirs(frames_sbs_dir, exist_ok=True)

    try:
        # 3. Video Info & Audio Extraction
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: 
            gr.Warning("Could not determine video FPS. Defaulting to 25. Output video might have incorrect speed.")
            fps = 25.0 
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 根据用户输入的剪切时间点，计算需要处理的起止帧索引
        clip_start_frame = 0
        clip_end_frame = frame_count  # 默认处理到视频结尾
        if start_time is not None and start_time > 0:
            clip_start_frame = int(start_time * fps)
        if end_time is not None and end_time > 0:
            clip_end_frame = int(end_time * fps)
        # 边界保护：确保起止帧在合法范围内
        if clip_start_frame < 0:
            clip_start_frame = 0
        if clip_end_frame > frame_count:
            clip_end_frame = frame_count
        if clip_start_frame >= clip_end_frame:
            gr.Error("开始时间必须小于结束时间，请检查输入。")
            return None
        target_frame_count = clip_end_frame - clip_start_frame
        
        temp_audio_path = os.path.join(work_parent_dir, "audio.aac")
        audio_extracted = False
        try:
            ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path]
            probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=False)
            if probe_result.returncode == 0 and probe_result.stdout.strip():
                # 构造音频提取命令，根据起止时间剪切音频（-ss/-to 放在输入前做快速 seek）
                cmd_extract_audio = ['ffmpeg', '-y']
                if start_time is not None and start_time > 0:
                    cmd_extract_audio += ['-ss', str(start_time)]
                if end_time is not None and end_time > 0:
                    cmd_extract_audio += ['-to', str(end_time)]
                cmd_extract_audio += ['-i', video_path, '-vn', '-acodec', 'copy', temp_audio_path]
                subprocess.run(cmd_extract_audio, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                audio_extracted = True
                print("Audio extracted successfully.")
            else:
                print(f"No audio stream found or ffprobe error. Probe output: {probe_result.stdout.strip()} {probe_result.stderr.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg error during audio extraction: {e.stderr.decode() if e.stderr else e.stdout.decode() if e.stdout else 'Unknown error'}")
        except FileNotFoundError:
            print("ffmpeg/ffprobe not found. Audio will not be processed. Please ensure ffmpeg is installed and in PATH.")

        # 4. 多线程流水线处理：抽帧 → GPU 推理 → SBS+落盘 三级流水线，让 CPU 与 GPU 并行工作
        print(f"Processing {target_frame_count} frames at {fps} FPS (pipeline, from frame {clip_start_frame} to {clip_end_frame})...")

        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # ===== 流水线配置 =====
        # 阶段1→2 队列：缓存放送 GPU 的待推理帧（PIL 图像，受内存约束不宜过大）
        GPU_QUEUE_SIZE = 2
        # 阶段2→3 队列：缓存放待落盘的 SBS 图像（SBS 图宽度为原图 2 倍，体积较大）
        SBS_QUEUE_SIZE = 8
        # SBS 线程池大小：仅并发 PNG 编码 + 落盘（纯 CPU/I/O，安全并发）
        SBS_WORKERS = min(4, (os.cpu_count() or 4))
        # 队列结束哨兵（用 object() 避免 None 等合法值冲突）
        _SENTINEL = object()

        gpu_queue = queue.Queue(maxsize=GPU_QUEUE_SIZE)
        sbs_queue = queue.Queue(maxsize=SBS_QUEUE_SIZE)
        # 工作线程异常容器：任意线程出错时记录，主线程检测后中止并向上抛
        worker_errors = []

        # ===== 阶段1：抽帧 + 预处理（CPU + I/O）=====
        def producer_fn():
            nonlocal target_frame_count  # 视频提前结束时需修正外层总数
            try:
                # 定位到起始帧，以便只读取需要剪切的范围
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
                for i in range(target_frame_count):
                    if worker_errors:  # 任意线程出错则提前结束
                        break
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Could only read {i} frames out of {target_frame_count}.")
                        target_frame_count = i  # 视频提前结束时修正总数
                        break
                    # OpenCV 读出为 BGR，转成 RGB 后构造 PIL 图像（等价于原来从 PNG 读取的效果）
                    input_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # 仅传 PIL 图像；normalize 与上 GPU 都集中在 GPU 线程做
                    gpu_queue.put((i, input_pil_image))
            except Exception as e:
                worker_errors.append(e)
            finally:
                cap.release()
                gpu_queue.put(_SENTINEL)  # 通知下游结束

        # ===== 阶段2：GPU 深度推理 + SBS 生成（单线程串行，避免多线程访问 GPU，兼容 MPS）=====
        def gpu_worker_fn():
            try:
                while True:
                    item = gpu_queue.get()
                    if item is _SENTINEL:
                        break
                    if worker_errors:
                        continue  # 出错后继续 drain 队列直到收到 SENTINEL，避免上游阻塞
                    i, input_pil_image = item
                    base_name = f"frame_{i:06d}"
                    # 深度推理：normalize + 上 GPU + 模型推理
                    image_tensor = transform_normalize(input_pil_image).unsqueeze(0).to(device=device, dtype=dtype)
                    depth_pil_image = process_depthmap_image(
                        depth_model, image_tensor, device, dtype, is_metric,
                        base_name, save_to_disk=False
                    )
                    # SBS 生成：内部也会用 GPU，与推理在同一线程串行，保证 GPU 单线程访问
                    sbs_pil_image = generate_sbs_image_from_depth(
                        input_pil_image, depth_pil_image, model_name,
                        sbs_method, sbs_depth_scale, sbs_mode, sbs_depth_blur_strength
                    )
                    sbs_queue.put((i, sbs_pil_image))
            except Exception as e:
                worker_errors.append(e)
            finally:
                sbs_queue.put(_SENTINEL)  # 通知下游结束

        # ===== 阶段3 单帧任务：SBS 图像落盘（在线程池里并行执行 PNG 编码 + I/O）=====
        def sbs_task(item):
            i, sbs_pil_image = item
            # 单帧失败不影响整体流程，捕获后跳过该帧
            try:
                if sbs_pil_image is not None:
                    base_name = f"frame_{i:06d}"
                    sbs_pil_image.save(os.path.join(frames_sbs_dir, f"sbs_{base_name}.png"))
                else:
                    gr.Warning(f"Failed to generate SBS for frame {i}. Skipping.")
            except Exception as e:
                print(f"Frame {i} save failed: {e}. Skipping.")
            return i

        # ===== 启动流水线（producer 与 gpu_worker 为后台线程，主线程消费 sbs_queue 协调进度）=====
        producer_thread = threading.Thread(target=producer_fn, name="frame-producer", daemon=True)
        gpu_thread = threading.Thread(target=gpu_worker_fn, name="gpu-worker", daemon=True)
        producer_thread.start()
        gpu_thread.start()

        actual_frames_processed = 0
        with ThreadPoolExecutor(max_workers=SBS_WORKERS, thread_name_prefix="sbs") as sbs_pool:
            pbar = progress.tqdm(total=target_frame_count, desc="Processing Frames")
            pending = []  # 已提交但未完成的 SBS future
            try:
                while True:
                    # 收割已完成的 future，及时更新进度（每轮先 drain 完成的，再取新任务）
                    for f in [x for x in pending if x.done()]:
                        f.result()  # 重抛任务异常（sbs_task 内部已 try，这里通常不会抛）
                        pending.remove(f)
                        pbar.update(1)
                        actual_frames_processed += 1
                    # 从 sbs_queue 取一项；带超时以便期间能收割已完成任务，避免进度条卡顿
                    try:
                        item = sbs_queue.get(timeout=0.2)
                    except queue.Empty:
                        if worker_errors:
                            break
                        continue
                    if item is _SENTINEL:
                        break
                    if worker_errors:
                        continue
                    pending.append(sbs_pool.submit(sbs_task, item))
                # 等待所有剩余 SBS 任务完成
                for fut in as_completed(pending):
                    fut.result()
                    pbar.update(1)
                    actual_frames_processed += 1
            finally:
                pbar.close()
                producer_thread.join(timeout=5)
                gpu_thread.join(timeout=5)

        # 工作线程异常向上抛（取第一个）
        if worker_errors:
            raise worker_errors[0]

        if actual_frames_processed == 0:
            gr.Error("No frames could be extracted from the video.")
            return None

        # 6. Assemble SBS Video
        sbs_frame_files = sorted([os.path.join(frames_sbs_dir, f) for f in os.listdir(frames_sbs_dir) if f.startswith("sbs_") and f.endswith(".png")])
        
        if not sbs_frame_files:
            gr.Error("No SBS frames were generated. Cannot create video.")
            return None

        sbs_video_no_audio_path = os.path.join(work_parent_dir, "sbs_video_no_audio.mp4")
        
        print(f"Assembling SBS video from {len(sbs_frame_files)} frames at {fps} FPS...")
        print("使用 VideoToolbox 硬件加速编码视频...")
        # 使用 VideoToolbox 硬件加速
        with imageio.get_writer(sbs_video_no_audio_path, fps=fps, codec='h264_videotoolbox', ffmpeg_params=['-b:v', '8M', '-pix_fmt', 'yuv420p']) as writer:
            for sbs_frame_file in progress.tqdm(sbs_frame_files, desc="Assembling Video"):
                writer.append_data(imageio.imread(sbs_frame_file))
        
        # 7. Add Audio Back (if extracted)
        if audio_extracted and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
            print(f"Adding audio back to video: {final_output_video_path}")
            cmd_mux = ['ffmpeg', '-y', '-i', sbs_video_no_audio_path, '-i', temp_audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-shortest', final_output_video_path]
            mux_result = subprocess.run(cmd_mux, capture_output=True, text=True, check=False)
            if mux_result.returncode != 0:
                print(f"ffmpeg error during audio muxing: {mux_result.stderr}")
                print("Falling back to video without audio.")
                shutil.move(sbs_video_no_audio_path, final_output_video_path)
            else:
                print("Audio muxed successfully.")
        else:
            print(f"Saving video without audio (or audio processing failed/not present): {final_output_video_path}")
            shutil.move(sbs_video_no_audio_path, final_output_video_path)
        
        print(f"Video processing complete. Output: {final_output_video_path}")
        return final_output_video_path

    except Exception as e:
        gr.Error(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 8. 保留中间文件（已保存到 output 目录），便于查看与排查
        if os.path.exists(work_parent_dir):
            print(f"中间文件已保存到: {work_parent_dir}")

# ================================================
# GRADIO UI
with gr.Blocks(title="SBS 2D To 3D") as demo:
    
    gr.Markdown("## SBS 2D To 3D Converter")

    gr.Markdown("输入视频文件路径进行处理（支持 MP4, AVI, MOV, MKV, TS 等格式）")
    with gr.Row():
        with gr.Column(scale=1):
            # 使用文本框直接输入视频文件路径
            video_input_component = gr.Textbox(
                label="视频文件路径",
                placeholder="请输入视频文件的完整路径，例如: /path/to/video.ts",
                lines=1
            )
            # 视频剪切：输入开始/结束时间（秒），留空或填0表示不剪切
            with gr.Group():
                gr.Markdown("#### 视频剪切（单位：秒，结束时间为 0 表示到视频结尾）")
                with gr.Row():
                    video_start_time = gr.Number(label="开始时间（秒）", value=0, minimum=0, step=0.1)
                    video_end_time = gr.Number(label="结束时间（秒）", value=0, minimum=0, step=0.1)
        with gr.Column(scale=1):
            model_dropdown_video = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                label="Select Model (for Depth Map)",
                value=AVAILABLE_MODELS[4] if len(AVAILABLE_MODELS) > 4 else (AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None)
            )

            with gr.Group():
                gr.Markdown("#### SBS 3D Parameters")
                with gr.Row():
                    sbs_method_video = gr.Dropdown(choices=["mesh_warping", "grid_sampling"], value="mesh_warping", label="SBS Method")       
                    sbs_mode_video = gr.Dropdown(choices=["parallel", "cross-eyed"], value="parallel", label="SBS View Mode")
                sbs_depth_scale_video = gr.Slider(minimum=1, maximum=150, value=40, step=1, label="SBS Depth Scale")
                sbs_depth_blur_strength_video = gr.Slider(minimum=1, maximum=15, value=7, step=2, label="SBS Depth Blur Strength (Odd Values)")
            process_video_button = gr.Button("Process SBS 3D Video", variant="primary")
    
    with gr.Row():
        output_sbs_video_component = gr.Video(label="Generated SBS 3D Video", interactive=False)

    # ========================================
    # EVENT HANDLERS
    # ========================================
    process_video_button.click(
        fn=generate_sbs_video,
        inputs=[
            video_input_component,
            model_dropdown_video,
            sbs_method_video,
            sbs_mode_video,       
            sbs_depth_scale_video,
            sbs_depth_blur_strength_video,
            video_start_time,
            video_end_time
        ],
        outputs=[output_sbs_video_component]
    )

if __name__ == "__main__":
    demo.launch()
