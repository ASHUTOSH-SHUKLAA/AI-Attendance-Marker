"""
Model Downloader Utility.
Downloads the official OpenCV face detection Haar Cascade model into `models/`.
Ensures the system operates independently without relying on python site-packages paths.
"""
import urllib.request
import ssl
from pathlib import Path
import config

CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
TARGET_FILE = config.MODELS_DIR / "haarcascade_frontalface_default.xml"


def download_cascade(force: bool = False) -> Path:
    """Downloads the official OpenCV Haar Cascade XML if not already present."""
    if TARGET_FILE.exists() and not force and TARGET_FILE.stat().st_size > 1000:
        return TARGET_FILE

    print(f"[Model Downloader] Downloading face detection model from:\n  {CASCADE_URL}")
    print(f"[Model Downloader] Target destination:\n  {TARGET_FILE}")

    # Bypass local SSL certificate errors if any exist in fresh python installs
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(
            CASCADE_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            content = response.read()

        TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TARGET_FILE, "wb") as f:
            f.write(content)

        print(f"[Model Downloader] Success! Downloaded {len(content):,} bytes to {TARGET_FILE.name}")
        return TARGET_FILE
    except Exception as e:
        print(f"[Model Downloader Error] Failed to download model: {e}")
        return TARGET_FILE


if __name__ == "__main__":
    download_cascade(force=True)
