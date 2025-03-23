import os
import subprocess
import requests
from tqdm import tqdm
import math

repo_url = "https://github.com/Mikubill/sd-webui-controlnet"
target_folder = r".\stable-diffusion-webui\extensions"
urls = [
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime_beta.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_canny.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_depth.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_canny.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_depth_midas.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_depth_zoe.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_lineart.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_openpose.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_diffusers_xl_sketch.safetensors",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth",
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth",
    "https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_style_sd14v1.pth",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_canny_full.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/diffusers_xl_depth_full.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ioclab_sd15_recolor.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sd15_plus.pth",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_xl.pth",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_canny_anime.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_depth_anime.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_openpose_anime.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_openpose_anime_v2.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_scribble_anime.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_canny_256lora.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_depth_256lora.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_recolor_256lora.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sai_xl_sketch_256lora.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sargezt_xl_depth.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sargezt_xl_depth_faid_vidit.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sargezt_xl_depth_zeed.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sargezt_xl_softedge.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_xl_canny.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_xl_openpose.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/t2i-adapter_xl_sketch.safetensors",
    "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose.safetensors",
	"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
	"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
	"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin",
	"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin",
	"https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"
]

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Change the current working directory to the target folder
os.chdir(target_folder)

print("Current Directory:", os.getcwd())

# Run the git clone command
subprocess.run(["git", "clone", repo_url])

print("ControlNet cloned successfully.")

# Change the current working directory to the cloned repository folder
repo_name = os.path.basename(repo_url)
 
os.chdir('sd-webui-controlnet')

print("Current Directory:", os.getcwd())
# Run the git clone command
subprocess.run(["git", "pull", repo_url])

print("ControlNet updated successfully.")

# Download the files into the models folder
models_folder = os.path.join(os.getcwd(), "models")
print("Current Directory:", os.getcwd())
print("models_folder Directory:", models_folder)
os.makedirs(models_folder, exist_ok=True)

total_files = len(urls)
downloaded_files = 0
print(f"Downloaded ({downloaded_files}/{total_files})")
for url in urls:
    # Extract the filename from the URL
    filename = os.path.basename(url)

    # Set the path for saving the file
    filepath = os.path.join(models_folder, filename)

    # Skip downloading if the file already exists and its size matches the expected size
    if os.path.exists(filepath):
        downloaded_files += 1
        print(f"Skipped {filename} Make sure that it was fully Downloaded! ({downloaded_files}/{total_files})")
        continue

    # Send a GET request to download the file
    response = requests.get(url, stream=True)

    # Get the total file size in bytes
    total_length = int(response.headers.get("content-length"))

    # Download the file in chunks and display progress
    with open(filepath, "wb") as file, tqdm(
        desc=f"Downloading {filename}", total=total_length, unit="B", unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(chunk_size=4096):
            file.write(data)
            progress_bar.update(len(data))

    downloaded_files += 1
    print(f"Downloaded {filename} ({downloaded_files}/{total_files})")

print("All files downloaded successfully.")
