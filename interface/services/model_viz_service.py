"""
Model visualization service for PyTorch models.

Provides architecture visualization using:
1. Simple text summary (no dependencies, always works)
2. torchinfo (detailed text summary, optional)
3. torchview (visual graph, optional)
"""
import torch
import torch.nn as nn
from io import StringIO
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile


class ModelVisualizationService:
    """Service for visualizing PyTorch model architectures."""

    @staticmethod
    def get_simple_summary(model: nn.Module) -> Dict[str, Any]:
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
        # Capture model architecture
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        print(model)
        sys.stdout = old_stdout
        architecture_str = buffer.getvalue()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
        size_mb = total_params * 4 / (1024**2)

        # Extract layer names
        layers = [name for name, _ in model.named_modules() if name]

        return {
            'architecture_str': architecture_str,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': size_mb,
            'layers': layers,
            'num_layers': len(layers)
        }

    @staticmethod
    def get_torchinfo_summary(
        model: nn.Module,
        input_size: tuple,
        device: str = 'cpu'
    ) -> Optional[str]:
        """
        Get detailed summary using torchinfo.

        Args:
            model: PyTorch model
            input_size: Input tensor size (batch_size, ...)
            device: Device to use for summary

        Returns:
            Summary string or None if torchinfo not available
        """
        try:
            from torchinfo import summary

            model_summary = summary(
                model,
                input_size=input_size,
                device=device,
                verbose=0,
                col_names=["input_size", "output_size", "num_params",
                          "kernel_size"],
                row_settings=["var_names"]
            )
            return str(model_summary)

        except ImportError:
            return None

    @staticmethod
    def get_torchview_graph(
        model: nn.Module,
        input_size: tuple,
        model_name: str = "Model",
        depth: int = 3
    ) -> Optional[Any]:
        """
        Generate visual graph using torchview.

        Args:
            model: PyTorch model
            input_size: Input tensor size
            model_name: Name for the graph
            depth: Depth of nested modules to show

        Returns:
            Graph object with .visual_graph attribute (graphviz.Digraph)
            or None if torchview not available
        """
        try:
            from torchview import draw_graph

            model_graph = draw_graph(
                model,
                input_size=input_size,
                device='meta',  # No actual computation
                expand_nested=True,
                graph_name=model_name,
                depth=depth,
                hide_inner_tensors=True,
                hide_module_functions=True
            )
            return model_graph

        except ImportError:
            return None

    @staticmethod
    def render_graph_to_png(
        graph_object: Any,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Render torchview graph to PNG file.

        Args:
            graph_object: Object returned by get_torchview_graph()
            output_path: Path to save PNG (without extension).
                        If None, uses temp file.

        Returns:
            Path to PNG file or None on failure
        """
        try:
            if output_path is None:
                tmp = tempfile.NamedTemporaryFile(
                    suffix='.png', delete=False
                )
                output_path = Path(tmp.name).with_suffix('')

            graph_object.visual_graph.render(
                filename=str(output_path),
                format='png',
                cleanup=True
            )
            return Path(f"{output_path}.png")

        except Exception:
            return None

    @staticmethod
    def get_layer_count_by_type(model: nn.Module) -> Dict[str, int]:
        """
        Count layers by type.

        Args:
            model: PyTorch model

        Returns:
            Dict mapping layer type to count
        """
        layer_counts = {}
        for module in model.modules():
            class_name = module.__class__.__name__
            layer_counts[class_name] = layer_counts.get(class_name, 0) + 1

        # Remove the model itself
        if model.__class__.__name__ in layer_counts:
            layer_counts[model.__class__.__name__] -= 1
            if layer_counts[model.__class__.__name__] == 0:
                del layer_counts[model.__class__.__name__]

        return layer_counts

    @staticmethod
    def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
        """
        Get detailed parameter statistics.

        Args:
            model: PyTorch model

        Returns:
            Dict with parameter stats by layer
        """
        stats = []
        for name, param in model.named_parameters():
            stats.append({
                'name': name,
                'shape': tuple(param.shape),
                'num_params': param.numel(),
                'trainable': param.requires_grad,
                'dtype': str(param.dtype)
            })
        return stats
