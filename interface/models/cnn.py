"""
CNN models: VGG16 and ResNet50 for spectrogram classification.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
import numpy as np

from .base import BaseModel


class VGG16FC(nn.Module):
    """VGG16 with frozen features and trainable classifier."""

    def __init__(self, num_classes: int, from_array: bool = False):
        super().__init__()
        self.from_array = from_array
        vgg = tv_models.vgg16(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(vgg.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(25088, num_classes)

    def forward(self, x):
        if self.from_array:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        elif x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class ResNet50FC(nn.Module):
    """ResNet50 with frozen features and trainable classifier."""

    def __init__(self, num_classes: int):
        super().__init__()
        resnet = tv_models.resnet50(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(100352, num_classes)

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class CNNModelAdapter(BaseModel):
    """Adapter to wrap PyTorch CNN models with BaseModel interface."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, x: np.ndarray) -> int:
        proba = self.predict_proba(x)
        return int(np.argmax(proba))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Validate input: must be RGB spectrogram (H, W, 3)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB spectrogram with shape (H, W, 3), got shape {x.shape}"
            )

        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            outputs = self.model(x_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        return proba

    def get_required_data_type(self) -> str:
        return 'spectrogram'
