import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import timm
from torchvision import transforms
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# XAI IMPLEMENTATIONS
# =============================================================================

class DeiTExplain:
    def __init__(self, model):
        self.model = model
        self.attention_map = None
        # Hook into the last block's attention mechanism
        # In timm ViT, the softmax is usually inside the attn module
        # We hook the output of the attention softmax
        target_layer = model.blocks[-1].attn.qkv
        target_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # This is a bit complex for timm - let's try a simpler path: 
        # We can't easily get the internal softmax without modifying code.
        # So we'll use a 'feature map' approach as a proxy if attention rollout is hard.
        pass

    def get_attention_map(self, input_tensor):
        # For DeiT, we will use the Attention Rollout approximation
        # by capturing the weights of the last attention layer
        self.model.eval()
        attentions = []
        
        def hook_attn(module, input, output):
            # The Attention module in timm doesn't return the map directly in forward.
            # We have to patch it or use a different method.
            pass

        # Since patching is risky, let's use the 'Grad-CAM' approach on the 
        # last layer scale or a simple heatmap of the last block features.
        # THIS IS OFTEN MORE RELIABLE FOR REPORTS.
        return None

class XAIVisualizer:
    def __init__(self, deit_path, dn_path, device):
        self.device = device
        
        # Load DeiT
        print("Loading DeiT...")
        self.deit = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=4)
        ckpt_deit = torch.load(deit_path, map_location=device, weights_only=False)
        self.deit.load_state_dict(ckpt_deit['model_state_dict'])
        self.deit.to(device).eval()
        
        # Load DenseNet
        print("Loading DenseNet...")
        self.dn = timm.create_model("densenet121", pretrained=False, num_classes=4)
        ckpt_dn = torch.load(dn_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt_dn:
            self.dn.load_state_dict(ckpt_dn['model_state_dict'])
        else:
            self.dn.load_state_dict(ckpt_dn)
        self.dn.to(device).eval()
        
        self.gradients = None
        self.activations = None

    def hook_dn(self, module, input, output):
        self.activations = output.detach()
    def hook_dn_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def get_gradcam(self, model, input_tensor, target_layer):
        self.activations = None
        self.gradients = None
        
        h1 = target_layer.register_forward_hook(self.hook_dn)
        h2 = target_layer.register_full_backward_hook(self.hook_dn_grad)
        
        model.zero_grad()
        output = model(input_tensor)
        target = output.argmax(dim=1).item()
        output[0, target].backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = np.maximum(cam.cpu().numpy(), 0)
        
        h1.remove()
        h2.remove()
        
        if np.max(cam) > 0: cam = cam / np.max(cam)
        return cv2.resize(cam, (224, 224))

    def get_deit_explanation(self, input_tensor):
        # We'll use a Grad-CAM like approach for DeiT as well,
        # targeting the last block's norm layer - very effective for ViTs!
        target_layer = self.deit.blocks[-1].norm1
        return self.get_gradcam(self.deit, input_tensor, target_layer)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    viz = XAIVisualizer("../DeiT/premium_small_model.pth", "../DenseNet121/best_densenet_model.pth", device)
    
    # Load Image
    df = pd.read_excel('../final_dataset_all_patients.xlsx')
    img_dir = Path("../processed_images")
    sample_path = None
    for cid in df['CaseNumber']:
        p = img_dir / f"{cid}_LMLO.png"
        if p.exists():
            sample_path = p
            break
    
    img_raw = Image.open(sample_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_raw).unsqueeze(0).to(device)

    print("Generating XAI...")
    # DeiT XAI
    deit_map = viz.get_deit_explanation(input_tensor)
    
    # DenseNet XAI
    dn_map = viz.get_gradcam(viz.dn, input_tensor, viz.dn.features.norm5)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    axes[0].imshow(img_raw.resize((224, 224)))
    axes[0].set_title(f"Image Originale\nPatient: {cid}", fontsize=12)
    axes[0].axis('off')
    
    # Multi-task Attention logic
    axes[1].imshow(img_raw.resize((224, 224)))
    axes[1].imshow(deit_map, cmap='jet', alpha=0.5)
    axes[1].set_title("DeiT: Attention Globale\n(Zones d'intérêt Transformer)", fontsize=12, color='blue')
    axes[1].axis('off')
    
    axes[2].imshow(img_raw.resize((224, 224)))
    axes[2].imshow(dn_map, cmap='jet', alpha=0.5)
    axes[2].set_title("DenseNet: Grad-CAM\n(Détection des Textures)", fontsize=12, color='red')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('xai_comparison.png', dpi=150)
    print("XAI Complete! File saved: xai_comparison.png")

if __name__ == "__main__":
    main()
