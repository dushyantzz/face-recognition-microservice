"""Download pretrained models for face recognition"""
import os
import urllib.request
from pathlib import Path
import argparse
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    """Download file with progress bar"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def download_models(models_dir: str = "models"):
    """Download pretrained models"""
    
    # Model URLs (these are placeholders - replace with actual URLs)
    models = {
        "retinaface_resnet50": {
            "url": "https://github.com/biubug6/Pytorch_Retinaface/releases/download/1.0/Resnet50_Final.pth",
            "path": os.path.join(models_dir, "detection", "retinaface_resnet50.pth")
        },
        "retinaface_mobilenet": {
            "url": "https://github.com/biubug6/Pytorch_Retinaface/releases/download/1.0/mobilenet0.25_Final.pth",
            "path": os.path.join(models_dir, "detection", "retinaface_mobilenet.pth")
        },
        # AdaFace models would need to be downloaded from their official repo
        # "adaface_ir101": {
        #     "url": "https://...",
        #     "path": os.path.join(models_dir, "embeddings", "adaface_ir101_webface12m.ckpt")
        # }
    }
    
    print("Downloading pretrained models...\n")
    
    for model_name, model_info in models.items():
        print(f"Downloading {model_name}...")
        
        if os.path.exists(model_info["path"]):
            print(f"  Model already exists: {model_info['path']}")
            continue
        
        try:
            download_file(model_info["url"], model_info["path"])
            print(f"  Downloaded: {model_info['path']}\n")
        except Exception as e:
            print(f"  Error downloading {model_name}: {e}\n")
    
    print("\nModel download complete!")
    print("\nNote: Some models may need to be downloaded manually from their official repositories:")
    print("  - AdaFace: https://github.com/mk-minchul/AdaFace")
    print("  - RetinaFace: https://github.com/biubug6/Pytorch_Retinaface")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save models")
    
    args = parser.parse_args()
    
    download_models(args.models_dir)


if __name__ == "__main__":
    main()