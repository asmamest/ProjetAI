import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration - DENSENET121 PREMIUM
# =============================================================================
class Config:
    # Look for data in the parent directory
    EXCEL_FILE = Path("../final_dataset_all_patients.xlsx")
    IMG_DIR = Path("../processed_images")
    
    # Model
    MODEL_NAME = "densenet121"
    NUM_CLASSES = 4
    IMG_SIZE = 224
    
    # Hyperparameters for Accuracy
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    
    # Data Split
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_SAVE_PATH = "best_densenet_model.pth"

if Config.DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# =============================================================================
# Dataset
# =============================================================================
class MammographyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        for idx, row in self.data.iterrows():
            for view in ['LCC', 'LMLO', 'RCC', 'RMLO']:
                img_path = img_dir / f"{row['CaseNumber']}_{view}.png"
                if img_path.exists():
                    self.samples.append({
                        'img_path': img_path,
                        'label': row['label'],
                        'case_number': row['CaseNumber']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        return image, label, sample['case_number']

# =============================================================================
# Advanced Data Augmentation (Kaggle-inspired)
# =============================================================================
def get_train_transforms():
    return A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.CLAHE(clip_limit=2.0),
        ], p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# =============================================================================
# Training Functions (Patient Level Voting)
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, targets, predictions = 0.0, [], []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(loader.dataset), accuracy_score(targets, predictions)

def validate_patient_level(model, loader, criterion, device):
    model.eval()
    patient_results, patient_targets = {}, {}
    
    with torch.no_grad():
        for images, labels, case_numbers in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for i, case_id in enumerate(case_numbers):
                cid = int(case_id.item()) if hasattr(case_id, 'item') else int(case_id)
                if cid not in patient_results:
                    patient_results[cid] = []
                    patient_targets[cid] = labels[i].item()
                patient_results[cid].append(probs[i])
                
    y_true, y_pred = [], []
    for cid in patient_results:
        final_prob = np.mean(patient_results[cid], axis=0)
        y_pred.append(np.argmax(final_prob))
        y_true.append(patient_targets[cid])
        
    return accuracy_score(y_true, y_pred), y_pred, y_true

# =============================================================================
# Main
# =============================================================================
def main():
    print("\n--- PREMIUM DENSENET121 TRAINING ---")
    
    # Load data from parent directory
    if not Config.EXCEL_FILE.exists():
        print(f"Error: {Config.EXCEL_FILE} not found. Please run from the DenseNet121 folder.")
        return

    df_raw = pd.read_excel(Config.EXCEL_FILE)
    print("Filtering images...")
    existing_files = set(f.name for f in Config.IMG_DIR.glob("*.png"))
    def has_images(c):
        return any(f"{c}_{v}.png" in existing_files for v in ['LCC', 'LMLO', 'RCC', 'RMLO'])
    
    df = df_raw[df_raw['CaseNumber'].apply(has_images)].copy()
    print(f"Found {len(df)} patients with processed images.")
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Breast_Density'])
    
    # Data Split (70/30 as requested)
    train_df, test_df = train_test_split(df, test_size=Config.TEST_SIZE, stratify=df['label'], random_state=Config.RANDOM_STATE)
    
    train_ds = MammographyDataset(train_df, Config.IMG_DIR, get_train_transforms())
    test_ds = MammographyDataset(test_df, Config.IMG_DIR, get_val_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True)
    
    model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE.type == 'cuda'))
    
    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, Config.DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Image Acc: {train_acc:.4f}")
        
        val_acc, _, _ = validate_patient_level(model, test_loader, criterion, Config.DEVICE)
        print(f"Patient-level Test Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'label_encoder': le}, Config.MODEL_SAVE_PATH)
            print(f"✓ New Best Patient Accuracy: {val_acc:.4f}")

    print(f"\nTraining Finished! Best Patient Acc (DenseNet121): {best_acc:.4f}")

if __name__ == "__main__":
    main()
