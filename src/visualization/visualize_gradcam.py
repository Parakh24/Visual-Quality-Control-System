"""
Grad-CAM Visualization Script for PCB Quality Control
"""
import sys
from pathlib import Path

#  Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.modeling.model_factory import ModelFactory
from src.common.config import ModelConfig, Paths
from src.visualization.grad_cam import GradCAM, overlay_heatmap


def main():
    """Main function to run Grad-CAM visualization"""
    
    print("=" * 60)
    print("Grad-CAM Visualization for PCB Quality Control")
    print("=" * 60)
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ==================== Load Model ====================
    print("\n[1] Loading Model...")
    
    # Choose model type: BASELINE_CNN, MOBILENETV2, or RESNET50
    model_type = ModelConfig.BASELINE_CNN  # Change as needed
    
    model = ModelFactory.create_model(
        model_type, 
        num_classes=2, 
        device=device
    )
    
    # Load trained weights if available
    weights_path = Paths.BASELINE_MODEL  # or Paths.TRANSFER_MODEL
    
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f" Loaded weights from: {weights_path}")
    else:
        print(f" No trained weights found at {weights_path}")
        print("   Using randomly initialized model for demo...")
    
    model.eval()
    
    # ==================== Load Image ====================
    print("\n[2] Loading Image...")
    
    image_path = PROJECT_ROOT / "data" / "splits" / "test" / "sample_pcb.jpg"
    
    if not image_path.exists():
        print(f"Image not found at {image_path}")
        print("   Creating dummy image for demo...")
        create_dummy_pcb_image(image_path)
    
    # Load and preprocess image
    img_pil = Image.open(image_path).convert("RGB")
    original_img = np.array(img_pil)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    print(f" Image loaded: {image_path}")
    print(f"   Input tensor shape: {input_tensor.shape}")
    
    # ==================== Get Target Layer ====================
    print("\n[3] Setting up Grad-CAM...")
    
    target_layer = get_target_layer(model, model_type)
    print(f"   Target layer: {target_layer}")
    
    # ==================== Generate Grad-CAM ====================
    print("\n[4] Generating Grad-CAM heatmap...")
    
    gradcam = GradCAM(model, target_layer, device=device)
    heatmap, predicted_class = gradcam.generate_heatmap(input_tensor)
    
    class_names = ["Defective", "Good"]  # Adjust based on your classes
    print(f"   Predicted Class: {predicted_class} ({class_names[predicted_class]})")
    
    # ==================== Overlay Heatmap ====================
    print("\n[5] Creating visualization...")
    
    # Resize original image to match heatmap
    original_resized = cv2.resize(original_img, (224, 224))
    
    # Overlay heatmap on original image
    overlayed_image = overlay_heatmap(original_resized, heatmap, alpha=0.5)
    
    # ==================== Display & Save Results ====================
    print("\n[6] Saving results...")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    axes[0].imshow(original_resized)
    axes[0].set_title("Original PCB Image", fontsize=14)
    axes[0].axis('off')
    
    # Heatmap Only
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14)
    axes[1].axis('off')
    
    # Overlayed Image
    axes[2].imshow(overlayed_image)
    axes[2].set_title(f"Overlay (Pred: {class_names[predicted_class]})", fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle("Grad-CAM Visualization for PCB Quality Control", fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_path = PROJECT_ROOT / "gradcam_result.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Result saved to: {output_path}")
    
    # Show figure
    plt.show()
    
    print("\n" + "=" * 60)
    print("Grad-CAM Visualization Complete!")
    print("=" * 60)


def get_target_layer(model, model_type):
    """
    Get the appropriate target layer based on model type
    
    Args:
        model: The neural network model
        model_type: Type of model (from ModelConfig)
        
    Returns:
        The target layer for Grad-CAM
    """
    if model_type == ModelConfig.BASELINE_CNN:
        return model.conv3
    elif model_type == ModelConfig.MOBILENETV2:
        return model.backbone.features[-1]
    elif model_type == ModelConfig.RESNET50:
        return model.backbone.layer4[-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_dummy_pcb_image(save_path):
    """
    Create a dummy PCB image for testing
    
    Args:
        save_path: Path to save the dummy image
    """
    # Create 224x224 RGB image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 40  # Dark gray background
    
    # Draw PCB-like patterns
    # Green PCB board color
    cv2.rectangle(img, (10, 10), (214, 214), (0, 100, 0), -1)
    
    # Copper traces (horizontal)
    for y in range(30, 200, 25):
        cv2.line(img, (20, y), (200, y), (0, 200, 200), 2)
    
    # Copper traces (vertical)
    for x in range(30, 200, 30):
        cv2.line(img, (x, 20), (x, 200), (0, 200, 200), 2)
    
    # Components (rectangles)
    cv2.rectangle(img, (50, 50), (90, 80), (50, 50, 50), -1)
    cv2.rectangle(img, (120, 100), (180, 140), (50, 50, 50), -1)
    cv2.rectangle(img, (40, 150), (80, 190), (50, 50, 50), -1)
    
    # Solder points (circles)
    for x in range(40, 200, 30):
        for y in range(40, 200, 30):
            cv2.circle(img, (x, y), 5, (180, 180, 180), -1)
    
    # Add a "defect" (red spot)
    cv2.circle(img, (150, 170), 12, (0, 0, 255), -1)
    
    # Save image
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f" Created dummy PCB image at: {save_path}")


if __name__ == "__main__":
    main()
