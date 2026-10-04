"""
Download required MediaPipe models
"""
import os
import urllib.request

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")

os.makedirs(_MODELS_DIR, exist_ok=True)

models = {
    "hand_landmarker": [
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task",
        "https://github.com/google-ai-edge/mediapipe/releases/download/v0.10.0/hand_landmarker.task",
    ]
}

_MIN_MODEL_SIZE = 1_000_000

_ZIP_MAGIC = b"PK"


def _is_valid_model(path: str) -> bool:
    """Return True if *path* looks like a valid binary model file."""
    try:
        size = os.path.getsize(path)
        if size < _MIN_MODEL_SIZE:
            return False
        with open(path, "rb") as f:
            magic = f.read(2)
        return magic == _ZIP_MAGIC
    except OSError:
        return False


for name, urls in models.items():
    model_path = os.path.join(_MODELS_DIR, f"{name}.task")

    if os.path.exists(model_path):
        if _is_valid_model(model_path):
            print(f"[OK] Model {name} already exists and is valid")
            continue
        else:
            print(f"[FAIL] Model {name} is corrupted or too small, re-downloading...")
            os.remove(model_path)

    downloaded = False
    for url in urls:
        print(f"Trying to download {name} from {url}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            if _is_valid_model(model_path):
                print(f"[OK] Downloaded {name}")
                downloaded = True
                break
            else:
                print(f"  Downloaded file is invalid, trying next source...")
                os.remove(model_path)
        except Exception as e:
            print(f"  Failed: {e}")

    if not downloaded:
        print(f"[FAIL] Could not download {name} from any source")
