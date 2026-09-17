"""
Download Kronos models from Hugging Face.

Run this script when you have unrestricted internet access (e.g., home network or VPN).
The models will be cached locally and used offline afterward.

Usage:
    python download_models.py
"""
import os
import sys

def main():
    # Disable SSL verification for corporate environments
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from huggingface_hub import snapshot_download

    models = [
        ("NeoQuasar/Kronos-Tokenizer-base", "Kronos Tokenizer"),
        ("NeoQuasar/Kronos-small", "Kronos Small (24.7M params)"),
    ]

    for repo_id, name in models:
        print(f"\n{'='*60}")
        print(f"Downloading: {name} ({repo_id})")
        print(f"{'='*60}")
        try:
            path = snapshot_download(repo_id)
            print(f"SUCCESS: Downloaded to {path}")
        except Exception as e:
            print(f"FAILED: {e}")
            print(f"\nIf CDN is blocked, try setting HF_ENDPOINT:")
            print(f"  set HF_ENDPOINT=https://hf-mirror.com")
            print(f"  python download_models.py")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("All models downloaded! You can now run predictions offline.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
