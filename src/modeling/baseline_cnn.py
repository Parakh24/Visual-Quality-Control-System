"""
Baseline CNN built from scratch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    A simple CNN architecture built from scratch for image classification
    
    Architecture:
    - Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool 
    - Conv2d -> ReLU -> MaxPool -> Flatten -> FC -> Dropout -> FC
    """
    
    def __init__(self, num_classes=2, input_channels=3):
        """
        Initialize the baseline CNN
        
        Args:
            num_classes (int): Number of output classes
            input_channels (int): Number of input channels (3 for RGB)
        """
        super(BaselineCNN, self).__init__()
        
        # Block 1: Conv + ReLU + MaxPool
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        
        # Block 2: Conv + ReLU + MaxPool
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        # Block 3: Conv + ReLU + MaxPool
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_model_summary(self):
        """Return model architecture summary"""
        summary = """
        ========== Baseline CNN Architecture ==========
        Input: (Batch, 3, 224, 224)
        
        Layer 1: Conv2d(3, 32) -> BatchNorm -> ReLU -> MaxPool2d(2,2)
        Layer 2: Conv2d(32, 64) -> BatchNorm -> ReLU -> MaxPool2d(2,2)
        Layer 3: Conv2d(64, 128) -> BatchNorm -> ReLU -> MaxPool2d(2,2)
        
        Global Average Pooling -> Flatten
        
        FC1: 128 -> 256 (ReLU + Dropout 0.5)
        FC2: 256 -> 128 (ReLU + Dropout 0.3)
        FC3: 128 -> 2 (Output)
        
        Total Parameters: ~1.5M
        =============================================
        """
        return summary


if __name__ == "__main__":
    # Test the model
    model = BaselineCNN(num_classes=2)
    print(model.get_model_summary())
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
