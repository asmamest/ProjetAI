import pandas as pd
from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_comparison():
    df = pd.read_excel('final_dataset_all_patients.xlsx')
    img_dir = Path('processed_images')
    
    # Target case
    case_id = None
    view = 'LMLO'
    
    for cid in df['CaseNumber']:
        dcm_path = Path(str(cid)) / f"{view}.dcm"
        png_path = img_dir / f"{cid}_{view}.png"
        
        if dcm_path.exists() and png_path.exists():
            case_id = cid
            break
            
    if not case_id:
        print("Could not find a case with both DICOM and PNG.")
        return

    # Load Raw DICOM
    dcm = pydicom.dcmread(Path(str(case_id)) / f"{view}.dcm")
    raw_img = dcm.pixel_array
    
    # Load Processed PNG
    proc_img = Image.open(img_dir / f"{case_id}_{view}.png")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Raw Image (Normalized for visualization)
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title(f"Image Brute (DICOM)\nCase: {case_id} {view}\nResolution: {raw_img.shape}")
    axes[0].axis('off')
    
    # Processed Image
    axes[1].imshow(proc_img, cmap='gray')
    axes[1].set_title(f"Image Traitée (PNG 512x512)\nNormalisée + Cropped + Squared")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_visualization.png', dpi=100)
    print(f"Comparison saved as comparison_visualization.png for Case {case_id}")

if __name__ == "__main__":
    create_comparison()
