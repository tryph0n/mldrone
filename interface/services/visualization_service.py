"""
Visualization service for creating Plotly figures.
Decouples plotting logic from UI.
"""
import numpy as np
import plotly.graph_objects as go
from typing import Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import COLORS, DRONE_CLASSES


class VisualizationService:
    """Creates visualization figures."""

    @staticmethod
    def create_psd_plot(psd_data: np.ndarray) -> go.Figure:
        """Create line plot for PSD features."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=psd_data,
            mode='lines',
            name='PSD',
            line=dict(color='#3498db', width=1.5)
        ))
        fig.update_layout(
            title='Power Spectral Density',
            xaxis_title='Frequency Bin',
            yaxis_title='PSD (normalized linear)',
            template='plotly_dark',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig

    @staticmethod
    def create_spectrogram_plot(spec_data: np.ndarray) -> go.Figure:
        """Create heatmap for spectrogram data."""
        # Handle different shapes: (H, W), (H, W, 1), or (H, W, 3)
        if spec_data.ndim == 3:
            if spec_data.shape[-1] == 3:
                display_data = np.mean(spec_data, axis=-1)
            else:
                display_data = spec_data[:, :, 0]
        else:
            display_data = spec_data

        fig = go.Figure(data=go.Heatmap(
            z=display_data,
            colorscale='Viridis',
            colorbar=dict(title='Intensity (normalized)')
        ))
        fig.update_layout(
            title='Spectrogram',
            xaxis_title='Time (ms)',
            yaxis_title='Frequency offset (MHz)',
            template='plotly_dark',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig

    @staticmethod
    def create_iq_plot(iq_data: np.ndarray) -> go.Figure:
        """Create dual line plot for IQ signal."""
        fig = go.Figure()

        n_samples = iq_data.shape[1] if iq_data.ndim == 2 else len(iq_data)

        x = np.arange(n_samples)
        i_data = iq_data[0] if iq_data.ndim == 2 else iq_data
        q_data = iq_data[1] if iq_data.ndim == 2 else None

        fig.add_trace(go.Scatter(
            x=x, y=i_data,
            mode='lines',
            name='In-phase (I)',
            line=dict(color='#e74c3c', width=1)
        ))

        if q_data is not None:
            fig.add_trace(go.Scatter(
                x=x, y=q_data,
                mode='lines',
                name='Quadrature (Q)',
                line=dict(color='#2ecc71', width=1)
            ))

        fig.update_layout(
            title='IQ Signal',
            xaxis_title='Sample',
            yaxis_title='Amplitude (normalized [0,1])',
            template='plotly_dark',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig

    @staticmethod
    def create_sample_plot(data_type: str, sample: np.ndarray) -> go.Figure:
        """Route to appropriate visualization based on data type."""
        plot_methods = {
            'psd': VisualizationService.create_psd_plot,
            'spectrogram': VisualizationService.create_spectrogram_plot,
            'iq': VisualizationService.create_iq_plot,
        }

        method = plot_methods.get(data_type)
        if not method:
            raise ValueError(f"Unknown data type: {data_type}")

        return method(sample)

    @staticmethod
    def create_probability_chart(
        model_name: str,
        probabilities: np.ndarray,
        class_names: np.ndarray = None
    ) -> go.Figure:
        """Create bar chart for class probabilities."""
        if class_names is None:
            class_names = DRONE_CLASSES

        fig = go.Figure(data=[
            go.Bar(
                x=class_names,
                y=probabilities * 100,
                marker_color=COLORS.get(model_name, '#888888'),
                text=[f"{p:.1f}%" for p in probabilities * 100],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title=f"{model_name} - Class Probabilities",
            xaxis_title="Class",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            template='plotly_dark',
            height=300
        )
        return fig

    @staticmethod
    def create_comparison_chart(
        results: list,
        class_names: np.ndarray = None
    ) -> go.Figure:
        """Create grouped bar chart comparing multiple models."""
        if class_names is None:
            class_names = DRONE_CLASSES

        fig = go.Figure()

        for result in results:
            fig.add_trace(go.Bar(
                name=result.model_name,
                x=class_names,
                y=result.probabilities * 100,
                marker_color=COLORS.get(result.model_name, '#888888'),
            ))

        fig.update_layout(
            title="Model Comparison - Class Probabilities",
            xaxis_title="Class",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            template='plotly_dark',
            height=400,
            barmode='group'
        )
        return fig
