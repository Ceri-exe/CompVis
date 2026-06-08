import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ConvBlock(nn.Module):
    """Dual convolutional layers with Batch Normalization and ReLU."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
        
class EncoderBlock(nn.Module):
    """Downsampling path: Extracts features and halves spatial dimensions."""
    def __init__(self, in_c: int, out_c: int): 
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, x: torch.Tensor):   
        skip_features = self.conv(x)
        pooled_features = self.pool(skip_features)
        return skip_features, pooled_features

class DecoderBlock(nn.Module):
    """Upsampling path: Recovers spatial dimensions and concatenates skip connections."""
    def __init__(self, in_c: int, out_c: int): 
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:], antialias=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    """U-Net architecture for binary semantic segmentation."""
    def __init__(self): 
        super().__init__()
        # Encoder
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.b = ConvBlock(512, 1024)
        
        # Decoder
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        
        # Final Segmentation Map Output
        self.out = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        b = self.b(p4) 
        
        u1 = self.d1(b, s4)
        u2 = self.d2(u1, s3)
        u3 = self.d3(u2, s2)
        u4 = self.d4(u3, s1)
        
        return self.out(u4)