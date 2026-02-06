"""
Factory pattern for model creation
"""
import os
import sys

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.common.config import ModelConfig
from src.modeling.baseline_cnn import BaselineCNN
from src.modeling.transfer_learning import (
    TransferLearningMobileNetV2,
    TransferLearningResNet50
)

class ModelFactory:
    """Factory for creating models"""
    
    @staticmethod
    def create_model(model_name, num_classes=2, device="cuda", **kwargs):
        """
        Create a model based on the specified name
        
        Args:
            model_name (str): Name of the model to create
            num_classes (int): Number of output classes
            device (str): Device to load model on ('cuda' or 'cpu')
            **kwargs: Additional arguments for specific models
            
        Returns:
            torch.nn.Module: The created model
            
        Raises:
            ValueError: If model_name is not recognized
        """
        
        model = None
        
        if model_name == ModelConfig.BASELINE_CNN:
            print(f"Creating {ModelConfig.BASELINE_CNN}...")
            model = BaselineCNN(num_classes=num_classes)
            
        elif model_name == ModelConfig.MOBILENETV2:
            print(f"Creating {ModelConfig.MOBILENETV2}...")
            freeze_backbone = kwargs.get('freeze_backbone', True)
            model = TransferLearningMobileNetV2(
                num_classes=num_classes,
                freeze_backbone=freeze_backbone
            )
            
        elif model_name == ModelConfig.RESNET50:
            print(f"Creating {ModelConfig.RESNET50}...")
            freeze_backbone = kwargs.get('freeze_backbone', True)
            model = TransferLearningResNet50(
                num_classes=num_classes,
                freeze_backbone=freeze_backbone
            )
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Move model to device
        model = model.to(device)
        
        # Print model info
        ModelFactory._print_model_info(model, model_name)
        
        return model
    
    @staticmethod
    def _print_model_info(model, model_name):
        """Print model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Frozen Parameters: {frozen_params:,}")
        print(f"Device: {next(model.parameters()).device}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    # Test all models
    models_to_test = [
        ModelConfig.BASELINE_CNN,
        ModelConfig.MOBILENETV2,
        ModelConfig.RESNET50
    ]
    
    for model_name in models_to_test:
        model = ModelFactory.create_model(model_name, num_classes=2, device="cpu")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to("cpu")
        output = model(dummy_input)
        print(f"Output shape: {output.shape}\n")
