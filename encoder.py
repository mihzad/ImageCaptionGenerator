import torch
import torch.nn as nn
from torchvision import models

class EncoderCNN(nn.Module):
    """
    Image feature extractor using pretrained MobileNetV3-large.
    Outputs a compact embedding vector for the decoder.
    """
    def __init__(self, embed_size):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(960, embed_size)   # 960 = MobileNetV3-large final channels
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            x = self.features(images)
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x