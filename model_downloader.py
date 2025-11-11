import os
import gdown

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "best_eligibility_model.pkl":
        "https://drive.google.com/uc?export=download&id=1Yn2YujwiErvHphboPIndWjoNygq1yErR",

    "best_max_emi_model.pkl":
        "https://drive.google.com/uc?export=download&id=1nJ3jai1SgddSF-eYOue6nchQKlYevb81",

    "eligibility_features.pkl":
        "https://drive.google.com/uc?export=download&id=1I7-Z8O4vL6DbyUnNrxdYNv-8Mfzy0Iir",

    "eligibility_scaler.pkl":
        "https://drive.google.com/uc?export=download&id=1-hMBWeW02_I_kxDi6CG0ImDllWOuL_eb",

    "emi_features.pkl":
        "https://drive.google.com/uc?export=download&id=1oiOqAZKmBPbC0Z-4CGKaDtFX-cTek5xP",

    "emi_scaler.pkl":
        "https://drive.google.com/uc?export=download&id=1brEBFqqCwDHhNP0291phWIBbvb4Aqq97",
}


def download_models():
    """Download each model file if not already present"""
    for filename, url in FILES.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            print(f"⬇ Downloading {filename}...")
            gdown.download(url, path, quiet=False)

    print("✅ All models downloaded successfully.")

