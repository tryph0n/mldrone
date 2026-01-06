"""Machine learning models for drone RF classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score


class PsdSVM:
    """SVM classifier for PSD features."""

    def __init__(self, C: float = 1.0, gamma: str = 'scale'):
        self.svc = svm.SVC(kernel='rbf', C=C, gamma=gamma)

    def fit(self, X, y):
        self.svc.fit(X, y)

    def predict(self, X):
        return self.svc.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1": f1_score(y, y_pred, average='weighted')
        }


class VGG16FC(nn.Module):
    """VGG16 with frozen features and trainable classifier."""

    def __init__(self, num_classes: int, from_array: bool = False):
        super().__init__()
        self.from_array = from_array

        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(vgg.children())[:-1])

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(25088, num_classes)

    def forward(self, x):
        # Handle grayscale arrays by repeating to 3 channels
        if self.from_array:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        elif x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def reset_weights(self):
        self.classifier.reset_parameters()


class ResNet50FC(nn.Module):
    """ResNet50 with frozen features and trainable classifier."""

    def __init__(self, num_classes: int):
        super().__init__()

        resnet = models.resnet50(weights='IMAGENET1K_V1')
        # Remove FC and adaptive pooling
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.features.parameters():
            param.requires_grad = False

        # Output of ResNet50 before pooling: 2048 x 7 x 7 = 100352
        self.classifier = nn.Linear(100352, num_classes)

    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def reset_weights(self):
        self.classifier.reset_parameters()


class RFUAVNet(nn.Module):
    """
    RF-UAV-Net: 1D CNN for raw IQ classification.

    Architecture (from IEEE paper):
    - R-Unit: Conv1d(2->64, k=5, s=5) + BN + ELU
    - 4x G-Units: GroupedConv1d(64->64, k=3, s=2, groups=8) + BN + ELU + skip
    - Multi-scale GAP: kernel sizes [1000, 500, 250, 125]
    - Dense: 320 -> num_classes

    Input: (batch, 2, 10000) - [real, imag] channels
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # R-unit
        self.conv_r = nn.Conv1d(2, 64, kernel_size=5, stride=5)
        self.bn_r = nn.BatchNorm1d(64)
        self.elu_r = nn.ELU()

        # G-units (4x)
        self.g_convs = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=3, stride=2, groups=8)
            for _ in range(4)
        ])
        self.g_bns = nn.ModuleList([nn.BatchNorm1d(64) for _ in range(4)])
        self.g_elus = nn.ModuleList([nn.ELU() for _ in range(4)])

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Multi-scale GAP
        self.gap1000 = nn.AvgPool1d(1000)
        self.gap500 = nn.AvgPool1d(500)
        self.gap250 = nn.AvgPool1d(250)
        self.gap125 = nn.AvgPool1d(125)

        # Classifier
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        # R-unit
        x = self.elu_r(self.bn_r(self.conv_r(x)))

        # G-units with residual connections
        g_outputs = []
        for i in range(4):
            g_out = self.g_elus[i](self.g_bns[i](self.g_convs[i](F.pad(x, (1, 0)))))
            g_outputs.append(g_out)
            x = g_out + self.pool(x)

        # Multi-scale GAP
        gaps = [
            self.gap1000(g_outputs[0]),
            self.gap500(g_outputs[1]),
            self.gap250(g_outputs[2]),
            self.gap125(g_outputs[3]),
            self.gap125(x)
        ]

        x = torch.cat(gaps, dim=1).flatten(start_dim=1)
        return self.fc(x)

    def reset_weights(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
