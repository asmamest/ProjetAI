import torch
import pandas as pd
from pathlib import Path

# Load the quick test model
checkpoint_path = "quick_test_deit_model.pth"
if Path(checkpoint_path).exists():
    try:
        # Using weights_only=False because the checkpoint contains a LabelEncoder object
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("--- Model Checkpoint Stats ---")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        if 'label_encoder' in checkpoint:
            print(f"Classes: {checkpoint['label_encoder'].classes_}")
        else:
            print("Label encoder not found in checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print(f"File {checkpoint_path} not found.")

# Check training script for potential issues
script_path = "train_deit_quick.py"
if Path(script_path).exists():
    with open(script_path, 'r') as f:
        content = f.read()
        if "num_workers=0" not in content.lower() and "NUM_WORKERS = 0" not in content:
            print("Warning: NUM_WORKERS might not be set to 0, which can cause issues on Windows.")
        if "pin_memory=True" in content and "cuda" not in content.lower():
             print("Warning: pin_memory=True used without CUDA.")

print("\n--- Data Check ---")
df = pd.read_excel('final_dataset_all_patients.xlsx')
img_dir = Path("processed_images")
missing_count = 0
found_count = 0
for idx, row in df.head(100).iterrows():
    for view in ['LCC', 'LMLO', 'RCC', 'RMLO']:
        p = img_dir / f"{row['CaseNumber']}_{view}.png"
        if p.exists():
            found_count += 1
        else:
            missing_count += 1

print(f"Checked 100 patients (400 possible images):")
print(f"Found: {found_count}")
print(f"Missing: {missing_count}")
