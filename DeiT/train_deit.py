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
    MODEL_NAME = "deit_base_patch16_224"  # DeiT model from timm
    NUM_CLASSES = 4  # A, B, C, D
    IMG_SIZE = 224
    
    # Training
    BATCH_SIZE = 16  # Reduced for better GPU memory usage
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Data Split
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15  # From remaining train data
    RANDOM_STATE = 42
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0  # Windows multiprocessing fix
    
    # Early Stopping
    PATIENCE = 10
    
    # Output
    MODEL_SAVE_PATH = "best_deit_density_model.pth"

print(f"Using device: {Config.DEVICE}")
if Config.DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# Dataset
# =============================================================================
class MammographyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, views=['LCC', 'LMLO', 'RCC', 'RMLO']):
        """
        Dataset for mammography images.
        For multi-view, we'll use all 4 views and average the predictions.
        For simplicity in this version, we'll use a single view per sample.
        """
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.views = views
        
        # Expand dataset: one row per view
        self.samples = []
        for idx, row in dataframe.iterrows():
            for view in views:
                img_path = img_dir / f"{row['CaseNumber']}_{view}.png"
                if img_path.exists():
                    self.samples.append({
                        'img_path': img_path,
                        'label': row['label'],
                        'patient_id': row['CaseNumber']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['img_path']).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return image, label

# =============================================================================
# Data Augmentation
# =============================================================================
def get_train_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

# =============================================================================
# Model
# =============================================================================
def create_model(num_classes=4):
    """Create DeiT model from timm."""
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=True,
        num_classes=num_classes
    )
    return model

# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(targets, predictions)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(targets, predictions)
    
    return epoch_loss, epoch_acc, predictions, targets

# =============================================================================
# Main Training Loop
# =============================================================================
def main():
    print("="*80)
    print("RSNA Breast Density Classification - DeiT Training")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_excel(Config.EXCEL_FILE)
    print(f"Total patients: {len(df)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Breast_Density'])
    print(f"Classes: {label_encoder.classes_}")
    print(f"Label distribution:\n{df['Breast_Density'].value_counts()}")
    
    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = train_test_split(
        df, 
        test_size=Config.TEST_SIZE, 
        stratify=df['label'],
        random_state=Config.RANDOM_STATE
    )
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=Config.VAL_SIZE,
        stratify=train_df['label'],
        random_state=Config.RANDOM_STATE
    )
    
    print(f"Train: {len(train_df)} patients")
    print(f"Val: {len(val_df)} patients")
    print(f"Test: {len(test_df)} patients")
    
    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = MammographyDataset(train_df, Config.IMG_DIR, get_train_transforms())
    val_dataset = MammographyDataset(val_df, Config.IMG_DIR, get_val_transforms())
    test_dataset = MammographyDataset(test_df, Config.IMG_DIR, get_val_transforms())
    
    print(f"Train samples (with views): {len(train_dataset)}")
    print(f"Val samples (with views): {len(val_dataset)}")
    print(f"Test samples (with views): {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE.type == 'cuda' else False
    )
    
    # Create model
    print(f"\n4. Creating model: {Config.MODEL_NAME}...")
    model = create_model(num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training
    print(f"\n5. Starting training for {Config.EPOCHS} epochs...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_encoder': label_encoder
            }, Config.MODEL_SAVE_PATH)
            print(f"✓ Best model saved! Val Acc: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{Config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for testing
    print(f"\n6. Loading best model for testing...")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print("\n7. Testing on test set...")
    test_loss, test_acc, test_preds, test_targets = validate_epoch(
        model, test_loader, criterion, Config.DEVICE
    )
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_targets,
        test_preds,
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_targets, test_preds)
    print(cm)
    
    print(f"\nModel saved to: {Config.MODEL_SAVE_PATH}")
    print("Training complete!")

if __name__ == "__main__":
    main()
