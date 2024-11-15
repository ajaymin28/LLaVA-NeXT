from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ajaymin28/vl-sg-AG-loras", 
    cache_dir="/root/LLaVA-NeXT/checkpoints"
)