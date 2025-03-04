import os
import subprocess
import requests
from tqdm import tqdm

# Constants
REPO_URL = "https://github.com/Mikubill/sd-webui-controlnet"
TARGET_FOLDER = r"/workspace/stable-diffusion-webui/extensions"
urls = [
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
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth"
]


def download_files(url_list, save_path):
    total_files = len(url_list)
    downloaded_files = 0
    print(f"Downloaded ({downloaded_files}/{total_files})")
    for url in url_list:
        filename = os.path.basename(url)
        filepath = os.path.join(save_path, filename)

        if os.path.exists(filepath):
            downloaded_files += 1
            print(
                f"Skipped {filename}. Make sure it was fully downloaded! ({downloaded_files}/{total_files})")
            continue

        response = requests.get(url, stream=True)
        total_length = int(response.headers.get("content-length"))
        with open(filepath, "wb") as file, tqdm(
            desc=f"Downloading {filename}", total=total_length, unit="B", unit_scale=True
        ) as progress_bar:
            for data in response.iter_content(chunk_size=4096):
                file.write(data)
                progress_bar.update(len(data))

        downloaded_files += 1
        print(f"Downloaded {filename} ({downloaded_files}/{total_files})")


if __name__ == "__main__":

    models_folder = "/workspace/ComfyUI/models/controlnet/1.5"
    os.makedirs(models_folder, exist_ok=True)

    download_files(urls, models_folder)

    print("All files downloaded successfully.")
