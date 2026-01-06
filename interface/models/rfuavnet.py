"""
RF-UAV-Net: 1D CNN for raw IQ signal classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseModel


class RFUAVNet(nn.Module):
    """RF-UAV-Net: 1D CNN for raw IQ classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_r = nn.Conv1d(2, 64, kernel_size=5, stride=5)
        self.bn_r = nn.BatchNorm1d(64)
        self.elu_r = nn.ELU()
        self.g_convs = nn.ModuleList([
            nn.Conv1d(64, 64, kernel_size=3, stride=2, groups=8)
            for _ in range(4)
        ])
        self.g_bns = nn.ModuleList([nn.BatchNorm1d(64) for _ in range(4)])
        self.g_elus = nn.ModuleList([nn.ELU() for _ in range(4)])
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap1000 = nn.AvgPool1d(1000)
        self.gap500 = nn.AvgPool1d(500)
        self.gap250 = nn.AvgPool1d(250)
        self.gap125 = nn.AvgPool1d(125)
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.elu_r(self.bn_r(self.conv_r(x)))
        g_outputs = []
        for i in range(4):
            g_out = self.g_elus[i](self.g_bns[i](self.g_convs[i](F.pad(x, (1, 0)))))
            g_outputs.append(g_out)
            x = g_out + self.pool(x)
        gaps = [
            self.gap1000(g_outputs[0]),
            self.gap500(g_outputs[1]),
            self.gap250(g_outputs[2]),
            self.gap125(g_outputs[3]),
            self.gap125(x)
        ]
        x = torch.cat(gaps, dim=1).flatten(start_dim=1)
        return self.fc(x)


class RFUAVNetAdapter(BaseModel):
    """Adapter for RFUAVNet with BaseModel interface."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, x: np.ndarray) -> int:
        proba = self.predict_proba(x)
        return int(np.argmax(proba))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            outputs = self.model(x_tensor)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        return proba

    def get_required_data_type(self) -> str:
        return 'iq'
