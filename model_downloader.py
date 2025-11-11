import os
import gdown

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DRIVE_URLS = {
    "best_eligibility_model.pkl": "https://drive.google.com/uc?export=download&id=1KRxxH8-TZRKRwcsm5SPM7sCaWsiw3PwA",
    "best_max_emi_model.pkl":     "https://drive.google.com/uc?export=download&id=1nJ3jai1SgddSF-eYOue6nchQKlYevb81",
    "eligibility_features.pkl":   "https://drive.google.com/uc?export=download&id=1I7-Z8O4vL6DbyUnNrxdYNv-8Mfzy0Iir",
    "eligibility_scaler.pkl":     "https://drive.google.com/uc?export=download&id=1-hMBWeW02_I_kxDi6CG0ImDllWOuL_eb",
    "emi_features.pkl":           "https://drive.google.com/uc?export=download&id=1oiOqAZKmBPbC0Z-4CGKaDtFX-cTek5xP",
    "emi_scaler.pkl":             "https://drive.google.com/uc?export=download&id=1brEBFqqCwDHhNP0291phWIBbvb4Aqq97",
}

def _download_file(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdown.download(url, out_path, quiet=False)

def ensure_one(basename: str, force: bool = False) -> str:
    """
    Ensure a single artifact exists locally; download if missing or if force=True.
    Returns the local path.
    """
    if basename not in DRIVE_URLS:
        raise KeyError(f"{basename!r} not found in DRIVE_URLS.")
    out = os.path.join(MODEL_DIR, basename)
    if force or not os.path.exists(out):
        print(f"⬇ Fetching: {basename}")
        _download_file(DRIVE_URLS[basename], out)
    else:
        print(f"✔ Exists: {basename}")
    return out

def download_models(force: bool = False) -> None:
    """Download all model artifacts. If force=True, re-download even if files exist."""
    for fname, url in DRIVE_URLS.items():
        out = os.path.join(MODEL_DIR, fname)
        if force or not os.path.exists(out):
            print(f"⬇ Downloading: {fname}")
            _download_file(url, out)
        else:
            print(f"✔ Already exists: {fname}")
    print("✅ All models processed.")
