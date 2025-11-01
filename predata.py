from huggingface_hub import snapshot_download
import os
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
# Download with parallel workers for faster download
snapshot_download(
    repo_id="facebook/map-anything",
    repo_type="dataset",
    local_dir="/home/xwh/project/map-anything/map-anything-dataset",
    max_workers=24,  # Adjust based on your connection and system
)