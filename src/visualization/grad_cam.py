"""
Grad-CAM Implementation for Model Interpretability
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Generates visual explanations for CNN-based models by highlighting
    important regions in the input image.
    """
    
    def __init__(self, model, target_layer, device='cuda'):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute Grad-CAM for
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._full_backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Save activations during forward pass"""
        self.activations = output.detach()
    
    def _full_backward_hook(self, module, grad_input, grad_output):
        """Save gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, uses predicted class)
            
        Returns:
            tuple: (heatmap, predicted_class_idx)
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        
        # Compute weights using global average pooling
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Compute weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam, class_idx


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image
    
    Args:
        img: Original image (H, W, 3) in RGB format
        cam: Class activation map (H, W), values in [0, 1]
        alpha: Transparency level (0.0 to 1.0)
        
    Returns:
        Overlayed image (H, W, 3) in RGB format
    """
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    # Normalize original image
    img_float = np.float32(img) / 255
    
    # Blend images
    overlayed = heatmap * alpha + img_float * (1 - alpha)
    overlayed = np.clip(overlayed, 0, 1)
    overlayed = np.uint8(255 * overlayed)
    
    return overlayed


def load_and_preprocess_image(image_path, image_size=(224, 224)):
    """
    Load and preprocess image for model input
    
    Args:
        image_path: Path to image file
        image_size: Target size (height, width)
        
    Returns:
        tuple: (original_image, input_tensor)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img)
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transform
    input_tensor = transform(img).unsqueeze(0)
    
    return original_img, input_tensor


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("Use GradCAM class to generate heatmaps.")
    print("Use overlay_heatmap() to visualize results.")
