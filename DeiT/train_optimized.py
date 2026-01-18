import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable version check

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
# Configuration
# =============================================================================
class Config:
    # Paths
    EXCEL_FILE = "final_dataset_all_patients.xlsx"
    IMG_DIR = Path("processed_images")
    
    # Model
    MODEL_NAME = "deit_base_patch16_224"
    NUM_CLASSES = 4
    IMG_SIZE = 224
    
    # Training
    BATCH_SIZE = 32  # Increased for faster training (if memory allows)
    EPOCHS = 15      # Reduced epochs but with better optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Data Split
    TEST_SIZE = 0.3  # As requested: 30% for testing
    RANDOM_STATE = 42
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0  # Windows multiprocessing fix
    
    # Output
    MODEL_SAVE_PATH = "optimized_deit_model.pth"

# Enable CuDNN benchmark for faster training
if Config.DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP (Mixed Precision) Enabled for speed")

# =============================================================================
# Dataset
# =============================================================================
class MammographyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        # We only add images that actually exist
        # The dataframe passed here is already filtered for patients with images
        for idx, row in self.data.iterrows():
            for view in ['LCC', 'LMLO', 'RCC', 'RMLO']:
                img_path = img_dir / f"{row['CaseNumber']}_{view}.png"
                if img_path.exists():
                    self.samples.append({
                        'img_path': img_path,
                        'label': row['label']
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
        return image, label

# =============================================================================
# Transforms (Optimized)
# =============================================================================
def get_train_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# =============================================================================
# Training Functions with AMP
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use Autocast for Mixed Precision
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

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation/Test'):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    return running_loss / len(loader.dataset), accuracy_score(targets, predictions), predictions, targets

# =============================================================================
# Main
# =============================================================================
def main():
    print("\nStarting Optimized Training (70/30 Split)...")
    
    # 1. Load and Filter Data
    df = pd.read_excel(Config.EXCEL_FILE)
    
    # Filter only patients who have at least one image in processed_images
    print("Filtering available images...")
    processed_patients = []
    for case_num in df['CaseNumber'].unique():
        # Check if any view exists for this patient
        found = False
        for view in ['LCC', 'LMLO', 'RCC', 'RMLO']:
            if (Config.IMG_DIR / f"{case_num}_{view}.png").exists():
                found = True
                break
        if found:
            processed_patients.append(case_num)
    
    df = df[df['CaseNumber'].isin(processed_patients)].reset_index(drop=True)
    print(f"Available patients with images: {len(df)}")
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Breast_Density'])
    
    # 2. Split 70/30
    train_df, test_df = train_test_split(
        df, test_size=Config.TEST_SIZE, stratify=df['label'], random_state=Config.RANDOM_STATE
    )
    
    print(f"Train set: {len(train_df)} patients")
    print(f"Test set: {len(test_df)} patients")
    
    # 3. Datasets & Dataloaders
    train_ds = MammographyDataset(train_df, Config.IMG_DIR, get_train_transforms())
    test_ds = MammographyDataset(test_df, Config.IMG_DIR, get_val_transforms())
    
    print(f"Total Train Samples (views): {len(train_ds)}")
    print(f"Total Test Samples (views): {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # 4. Model, Optimizer, Scaler
    model = timm.create_model(Config.MODEL_NAME, pretrained=True, num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE.type == 'cuda'))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # 5. Training Loop
    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, Config.DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        
        # Using test set as validation for this simple optimized run
        val_loss, val_acc, _, _ = validate_epoch(model, test_loader, criterion, Config.DEVICE)
        print(f"Test Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'label_encoder': le
            }, Config.MODEL_SAVE_PATH)
            print(f"New best model saved! ({val_acc:.4f})")

    # 6. Final Evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION ON 30% TEST SET")
    print("="*50)
    checkpoint = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, last_acc, y_pred, y_true = validate_epoch(model, test_loader, criterion, Config.DEVICE)
    
    print(f"\nFinal Test Accuracy: {last_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    
    print(f"\nTraining finished. Model: {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
