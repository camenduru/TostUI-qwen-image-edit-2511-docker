import os, shutil, requests, random, time, uuid, boto3, runpod
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime

import torch
import numpy as np
from PIL import Image

def download_file(url, save_dir, file_name, overwrite=True):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    if os.path.exists(file_path) and not overwrite:
        return file_path
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

import comfy
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_qwen, nodes_cfg, nodes_model_advanced, nodes_flux

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
ConditioningZeroOut = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
ModelSamplingAuraFlow = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("qwen_image_edit_2511_fp8_e4m3fn.safetensors", "default")[0]
    model = LoraLoaderModelOnly.load_lora_model_only(unet, "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", strength_model=1.0)[0]
    clip = CLIPLoader.load_clip("qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image")[0]
    vae = VAELoader.load_vae("qwen_image_vae.safetensors")[0]

@torch.inference_mode()
def generate(input):
    try:
        tmp_dir="/content/ComfyUI/output"
        os.makedirs(tmp_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:6]
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        s3_access_key_id = os.getenv('s3_access_key_id')
        s3_secret_access_key = os.getenv('s3_secret_access_key')
        s3_endpoint_url = os.getenv('s3_endpoint_url')
        s3_region_name = os.getenv('s3_region_name')
        s3_bucket_name = os.getenv('s3_bucket_name')
        s3_bucket_folder = os.getenv('s3_bucket_folder')
        s3 = boto3.client('s3', aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_access_key, endpoint_url=s3_endpoint_url, region_name=s3_region_name)

        values = input["input"]
        job_id = values['job_id']

        input_image1 = values['input_image1']
        input_image1 = download_file(url=input_image1, save_dir=tmp_dir, file_name='input_image1')
        input_image2 = values.get('input_image2')
        if input_image2:
            input_image2 = download_file(url=input_image2, save_dir=tmp_dir, file_name='input_image2')
        else:
            input_image2 = None
        input_image3 = values.get('input_image3')
        if input_image3:
            input_image3 = download_file(url=input_image3, save_dir=tmp_dir, file_name='input_image3')
        else:
            input_image3 = None

        prompt = values['prompt']
        seed = values['seed'] # 0
        steps = values['steps'] # 8
        cfg = values['cfg'] # 1.0
        sampler_name = values['sampler_name'] # er_sde
        scheduler = values['scheduler'] # beta
        denoise = values['denoise'] # 1.0
        shift = values['shift'] # 3.0
        strength = values['strength'] # 1.0

        custom_size = values['custom_size'] # True

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)

        input_image1 = LoadImage.load_image(input_image1)[0]
        input_image2 = LoadImage.load_image(input_image2)[0] if input_image2 else None
        input_image3 = LoadImage.load_image(input_image3)[0] if input_image3 else None
        positive = nodes_qwen.TextEncodeQwenImageEditPlus.execute(clip=clip, prompt=prompt, vae=vae, image1=input_image1, image2=input_image2, image3=input_image3)[0]
        if custom_size:
            width = values['width'] # 1024
            height = values['height'] # 1024
            batch_size = values['batch_size'] # 1.0
            latent_out = EmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]
        else:
            latent_out = VAEEncode.encode(vae=vae, pixels=input_image1)[0]
        positive = nodes_flux.FluxKontextMultiReferenceLatentMethod.execute(positive, "index_timestep_zero")[0]
        negative = ConditioningZeroOut.zero_out(positive)[0]
        model_shift = ModelSamplingAuraFlow.patch_aura(model, shift)[0]
        model_cfg = nodes_cfg.CFGNorm.execute(model_shift, strength)[0]
        samples = KSampler.sample(model_cfg, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_out, denoise=denoise)[0]
        comfy.model_management.unload_all_models()
        decoded = VAEDecode.decode(vae, samples)[0].detach()
        Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(f"{tmp_dir}/qwen_image_edit_2511.png")
        comfy.model_management.unload_all_models()

        result = f"{tmp_dir}/qwen_image_edit_2511.png"
        
        s3_key =  f"{s3_bucket_folder}/qwen_image_edit_2511-{current_time}-{seed}-{unique_id}.png"
        s3.upload_file(result, s3_bucket_name, s3_key, ExtraArgs={'ContentType': 'image/png'})
        result_url = f"{s3_endpoint_url}/{s3_bucket_name}/{s3_key}"

        return {"job_id": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        return {"job_id": job_id, "result": str(e), "status": "FAILED"}
    finally:
        directory_path = Path(tmp_dir)
        if directory_path.exists():
            shutil.rmtree(directory_path)
            print(f"Directory {directory_path} has been removed successfully.")
        else:
            print(f"Directory {directory_path} does not exist.")

runpod.serverless.start({"handler": generate})