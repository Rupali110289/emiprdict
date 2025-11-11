import gdown
import os

# ✅ Google Drive model file links  
DRIVE_LINKS = {
    "best_eligibility_model.pkl": "https://drive.google.com/uc?id=1Yn2YujwiErvHphboPIndWjoNygq1yErR",
    "best_max_emi_model.pkl": "https://drive.google.com/uc?id=1nJ3jai1SgddSF-eYOue6nchQKlYevb81",
    "eligibility_features.pkl": "https://drive.google.com/uc?id=1I7-Z8O4vL6DbyUnNrxdYNv-8Mfzy0Iir",
    "eligibility_scaler.pkl": "https://drive.google.com/uc?id=1-hMBWeW02_I_kxDi6CG0ImDllWOuL_eb",
    "emi_features.pkl": "https://drive.google.com/uc?id=1oiOqAZKmBPbC0Z-4CGKaDtFX-cTek5xP",
    "emi_scaler.pkl": "https://drive.google.com/uc?id=1brEBFqqCwDHhNP0291phWIBbvb4Aqq97",
}

def download_models():
    """
    Downloads all required model + scaler files from Google Drive.
    If files already exist, they won't be downloaded again.
    """
    os.makedirs("models", exist_ok=True)

    for filename, url in DRIVE_LINKS.items():
        out_path = os.path.join("models", filename)
        
        if not os.path.exists(out_path):
            try:
                gdown.download(url, out_path, quiet=False)
                print(f"✅ Downloaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"✅ Already exists: {filename}")
