import os
import time
import gdown
from typing import Optional

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Direct-download URLs (already converted to uc?id=...)
DRIVE_URLS = {
    "best_eligibility_model.pkl": "https://drive.google.com/uc?export=download&id=1Yn2YujwiErvHphboPIndWjoNygq1yErR",
    "best_max_emi_model.pkl":     "https://drive.google.com/uc?export=download&id=1nJ3jai1SgddSF-eYOue6nchQKlYevb81",
    "eligibility_features.pkl":   "https://drive.google.com/uc?export=download&id=1I7-Z8O4vL6DbyUnNrxdYNv-8Mfzy0Iir",
    "eligibility_scaler.pkl":     "https://drive.google.com/uc?export=download&id=1-hMBWeW02_I_kxDi6CG0ImDllWOuL_eb",
    "emi_features.pkl":           "https://drive.google.com/uc?export=download&id=1oiOqAZKmBPbC0Z-4CGKaDtFX-cTek5xP",
    "emi_scaler.pkl":             "https://drive.google.com/uc?export=download&id=1brEBFqqCwDHhNP0291phWIBbvb4Aqq97",
}

# Very small files are likely HTML error pages. Threshold keeps us safe.
_MIN_BYTES = {
    "best_eligibility_model.pkl": 200_000,
    "best_max_emi_model.pkl":     200_000,
    "eligibility_features.pkl":   500,       # small metadata is ok
    "eligibility_scaler.pkl":     500,
    "emi_features.pkl":           500,
    "emi_scaler.pkl":             500,
}

def _download(url: str, dst: str) -> None:
    gdown.download(url, dst, quiet=False)

def ensure_one(filename: str, force: bool = False) -> str:
    """
    Ensure a single file exists & is not obviously corrupted.
    Returns absolute path. Set force=True to re-download.
    """
    path = os.path.join(MODEL_DIR, filename)
    if force and os.path.exists(path):
        try: os.remove(path)
        except: pass

    # Try up to 3 times
    for attempt in range(1, 4):
        if not os.path.exists(path):
            _download(DRIVE_URLS[filename], path)

        # Validate size
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            size = 0

        if size >= _MIN_BYTES.get(filename, 500):
            return path  # looks fine

        # Otherwise retry: remove and sleep briefly
        try: os.remove(path)
        except: pass
        time.sleep(1.0)

    # Last resort: leave file (for debugging), but return path anyway
    return path

def download_all(force: bool = False) -> None:
    for name in DRIVE_URLS:
        ensure_one(name, force=force)
