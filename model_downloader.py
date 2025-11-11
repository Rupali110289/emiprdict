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

def download_models():
    for filename, url in DRIVE_URLS.items():
        out_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(out_path):
            print(f"⬇ Downloading {filename}...")
            gdown.download(url, out_path, quiet=False)

