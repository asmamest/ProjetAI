import torch
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load data from parent directory
df = pd.read_excel('../final_dataset_all_patients.xlsx')
img_dir = Path("../processed_images")

# Mock the LabelEncoder as done in the script
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Breast_Density'])

print(f"Total entries in Excel: {len(df)}")

# Check how many images exist in total
all_images = list(img_dir.glob("*.png"))
print(f"Total PNG images in processed_images: {len(all_images)}")

# Check class distribution of images that actually exist
existing_labels = []
samples_details = []
for idx, row in df.iterrows():
    for view in ['LCC', 'LMLO', 'RCC', 'RMLO']:
        p = img_dir / f"{row['CaseNumber']}_{view}.png"
        if p.exists():
            existing_labels.append(row['Breast_Density'])
            samples_details.append(row['label'])

if existing_labels:
    print("\n--- Distribution of existing processed images ---")
    dist = pd.Series(existing_labels).value_counts()
    print(dist)
    print(f"Total samples available: {len(existing_labels)}")
else:
    print("No images found matching Excel CaseNumbers.")

# Load the model to see the state_dict keys (check if it's really the right model)
checkpoint_path = "quick_test_deit_model.pth"
if Path(checkpoint_path).exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"\nModel checkpoint was for epoch: {checkpoint.get('epoch', 'N?A')}")
    print(f"Final Val Accuracy reported: {checkpoint.get('val_acc', 'N?A')}")

# Check if there was any error during the last run by checking for common temp files or logs
# (None found so far)

# Check the training script for potential logic errors
script_path = "train_deit_quick.py"
with open(script_path, 'r') as f:
    lines = f.readlines()
    # Check for potential issues
    for i, line in enumerate(lines):
        if "Config.QUICK_SAMPLE" in line:
            print(f"Line {i+1}: {line.strip()}")
        if "EPOCHS =" in line:
            print(f"Line {i+1}: {line.strip()}")
