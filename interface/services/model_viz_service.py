"""Model visualization service for PyTorch models."""

import sys
from io import StringIO
from typing import Any

import torch.nn as nn


class ModelVisualizationService:
    """Service for visualizing PyTorch model architectures."""

    @staticmethod
    def get_simple_summary(model: nn.Module) -> dict[str, Any]:
        """
        Get basic model summary without external dependencies.

        Args:
            model: PyTorch model

        Returns:
            dict with keys:
                - architecture_str: Full model architecture as string
                - total_params: Total number of parameters
                - trainable_params: Number of trainable parameters
                - size_mb: Model size in MB (assuming float32)
                - layers: List of layer names
        """
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        print(model)
        sys.stdout = old_stdout
        architecture_str = buffer.getvalue()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = total_params * 4 / (1024**2)

        layers = [name for name, _ in model.named_modules() if name]

        return {
            "architecture_str": architecture_str,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "size_mb": size_mb,
            "layers": layers,
            "num_layers": len(layers),
        }
