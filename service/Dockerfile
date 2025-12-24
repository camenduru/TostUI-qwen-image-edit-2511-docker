FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libegl1 libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev cmake && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg xvfb && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run -d /content -o cuda_12.9.1_575.57.08_linux.run && sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    rm -f /content/cuda_12.9.1_575.57.08_linux.run

USER camenduru

RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir xformers --extra-index-url https://download.pytorch.org/whl/cu129 && \
    git clone --depth 1 --branch master https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && cd /content/ComfyUI && pip install -r requirements.txt && \
    pip install opencv-python imageio imageio-ffmpeg ffmpeg-python moviepy timm scikit-image matplotlib diffusers accelerate peft torch-scatter runpod boto3 && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/xms991/Qwen-Image-Edit-2511-fp8-e4m3fn/resolve/main/qwen_image_edit_2511_fp8_e4m3fn.safetensors -d /content/ComfyUI/models/diffusion_models -o qwen_image_edit_2511_fp8_e4m3fn.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors -d /content/ComfyUI/models/clip -o qwen_2.5_vl_7b_fp8_scaled.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors -d /content/ComfyUI/models/vae -o qwen_image_vae.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors -d /content/ComfyUI/models/loras -o Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py