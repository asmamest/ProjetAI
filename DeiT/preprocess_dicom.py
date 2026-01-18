
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path("..") # Parent directory where data is located
EXCEL_FILE = BASE_DIR / "final_dataset_all_patients.xlsx"
OUTPUT_DIR = BASE_DIR / "processed_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = ['LCC', 'LMLO', 'RCC', 'RMLO']

# =============================================================================
# Preprocessing Functions
# =============================================================================

def load_dicom_raw(patient_id, view_name, base_dir):
    dcm_path = base_dir / str(patient_id) / f"{view_name}.dcm"
    
    # Check both string and int patient_id just in case
    if not dcm_path.exists():
        return None
        
    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(np.float32)
    except Exception:
        return None

    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    img = img * slope + intercept
    
    photometric = getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = np.max(img) - img
        
    return img

def window_image(img, low=5, high=99.5):
    if img is None or img.size == 0: return None
    low_val, high_val = np.percentile(img, [low, high])
    
    if high_val - low_val < 1e-6: 
        if img.max() - img.min() > 1e-6:
            img = (img - img.min()) / (img.max() - img.min())
        return img

    img = np.clip(img, low_val, high_val)
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img

def crop_to_breast(img, threshold=0.05):
    if img is None: return None
    mask = img > threshold
    if not mask.any(): return img
    
    y_any = mask.any(axis=1)
    x_any = mask.any(axis=0)
    
    y_min, y_max = np.where(y_any)[0][[0, -1]]
    x_min, x_max = np.where(x_any)[0][[0, -1]]
    
    return img[y_min:y_max + 1, x_min:x_max + 1]

def make_square(img):
    if img is None: return None
    h, w = img.shape
    if h == w: return img
    
    size = max(h, w)
    new_img = np.zeros((size, size), dtype=img.dtype)
    
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    
    new_img[y_off:y_off+h, x_off:x_off+w] = img
    return new_img

def to_pil_512(img, size=512):
    if img is None: return None
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
    return pil_img

# =============================================================================
# Worker Function for Parallel Processing
# =============================================================================
def process_patient(patient_id):
    """Processes all views for a single patient."""
    count_ok = 0
    count_err = 0
    
    for view in VIEWS:
        save_name = f"{patient_id}_{view}.png"
        save_path = OUTPUT_DIR / save_name
        
        # Skip if already exists (optimization for restarting)
        if save_path.exists():
            count_ok += 1
            continue

        try:
            # 1. Load
            img = load_dicom_raw(patient_id, view, BASE_DIR)
            if img is None:
                continue
            
            # 2. Process
            img = window_image(img)
            img = crop_to_breast(img)
            img = make_square(img)
            pil_img = to_pil_512(img, size=512)
            
            # 3. Save
            if pil_img:
                pil_img.save(save_path)
                count_ok += 1
        except Exception:
            count_err += 1
            
    return count_ok, count_err

# =============================================================================
# Main Layout
# =============================================================================
def main():
    # Use 75% of available cores to avoid freezing the system
    max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    print(f"Reading Excel file...")
    try:
        df = pd.read_excel(EXCEL_FILE)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    print(f"Found {len(df)} patients in Excel.")
    
    # Sample 60%
    df = df.sample(frac=0.6, random_state=42).reset_index(drop=True)
    print(f"Keeping 60% of data: {len(df)} patients to process.")
    
    patient_ids = df['CaseNumber'].tolist()
    
    print(f"Starting parallel processing with {max_workers} workers...")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    total_processed = 0
    total_errors = 0
    
    # ProcessPoolExecutor for parallel CPU tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map returns an iterator, we iterate it with tqdm
        results = list(tqdm(executor.map(process_patient, patient_ids), total=len(patient_ids)))
        
    for ok, err in results:
        total_processed += ok
        total_errors += err

    print("\nDone!")
    print(f"Processed images: {total_processed}")
    print(f"Errors: {total_errors}")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
