"""
Inference view - Select test samples and run inference.
Uses test samples from media/test_samples/{CLEAN,BOTH}/
"""
import sys
import time
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
import plotly.graph_objects as go

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import (
    TEST_SAMPLES_DIR, MODELS_DIR, DEVICE,
    DRONE_CLASSES, DRONE_INFO, INTERFERENCE_TYPES,
    DATA_TYPE_CONFIG, MODEL_PATHS, COLORS, MODEL_INPUT_SIZES
)
from models.rfuavnet import RFUAVNet, RFUAVNetAdapter
from models.cnn import VGG16FC, ResNet50FC, CNNModelAdapter
from models.svm import SVMAdapter
from services.model_viz_service import ModelVisualizationService


# =============================================================================
# Data Loading
# =============================================================================

def list_available_interferences() -> list[str]:
    """List available interference types (directories in test_samples)."""
    if not TEST_SAMPLES_DIR.exists():
        return []
    return [d.name for d in TEST_SAMPLES_DIR.iterdir()
            if d.is_dir() and d.name in INTERFERENCE_TYPES]


def list_available_files(interference: str) -> list[Path]:
    """List all NPZ files for an interference type."""
    base = TEST_SAMPLES_DIR / interference
    if not base.exists():
        return []
    return sorted(base.glob('*.npz'))


def parse_filename(filepath: Path) -> dict:
    """Parse filename to extract metadata.

    Formats:
    - psd_CLEAN_AIR_20.npz
    - spectrogram_CLEAN_AIR_224x224x3.npz
    - iq_CLEAN_AIR_20.npz
    """
    stem = filepath.stem
    parts = stem.split('_')
    if len(parts) >= 3:
        return {
            'data_type': parts[0],
            'interference': parts[1],
            'drone': parts[2],
            'suffix': '_'.join(parts[3:]) if len(parts) > 3 else ''
        }
    return {}


def list_available_types(interference: str) -> list[str]:
    """List available data types for an interference."""
    files = list_available_files(interference)
    types = set()
    for f in files:
        meta = parse_filename(f)
        if meta.get('data_type') in DATA_TYPE_CONFIG:
            types.add(meta['data_type'])
    return sorted(types)


def list_available_drones(interference: str, data_type: str) -> list[str]:
    """List available drones for interference/type."""
    files = list_available_files(interference)
    drones = []
    for f in files:
        meta = parse_filename(f)
        if meta.get('data_type') == data_type:
            drone = meta.get('drone')
            if drone and drone not in drones:
                drones.append(drone)
    return sorted(drones)


def get_sample_filepath(interference: str, data_type: str, drone: str) -> Optional[Path]:
    """Find the sample file for given parameters."""
    files = list_available_files(interference)
    for f in files:
        meta = parse_filename(f)
        if meta.get('data_type') == data_type and meta.get('drone') == drone:
            return f
    return None


@st.cache_data
def load_sample_file(filepath: str) -> Optional[dict]:
    """Load NPZ file and return contents."""
    try:
        data = np.load(filepath, allow_pickle=True)
        result = {'X': data['X'], 'y': data['y']}
        # Optional fields
        for key in ['y_interference', 'drone_class', 'interference_class']:
            if key in data:
                result[key] = data[key]
        return result
    except Exception as e:
        st.error(f"Failed to load {filepath}: {e}")
        return None


# =============================================================================
# Model Loading
# =============================================================================

@st.cache_resource
def load_rfuavnet():
    """Load RFUAVNet model."""
    model_path = MODEL_PATHS['RFUAVNet']
    if not model_path.exists():
        return None
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    num_classes = len(DRONE_CLASSES)
    model = RFUAVNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return RFUAVNetAdapter(model, DEVICE)


@st.cache_resource
def load_cnn(model_name: str):
    """Load VGG16 or ResNet50 model."""
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        return None
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    num_classes = len(DRONE_CLASSES)
    if model_name == 'VGG16':
        model = VGG16FC(num_classes=num_classes)
    else:
        model = ResNet50FC(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return CNNModelAdapter(model, DEVICE)


@st.cache_resource
def load_svm():
    """Load SVM model."""
    model_path = MODEL_PATHS['SVM']
    if not model_path.exists():
        return None
    with open(model_path, 'rb') as f:
        svm_data = pickle.load(f)
    # Handle both dict format and direct model
    model = svm_data.get('model', svm_data) if isinstance(svm_data, dict) else svm_data
    return SVMAdapter(model)


def load_model(model_name: str):
    """Load model by name."""
    if model_name == 'RFUAVNet':
        return load_rfuavnet()
    elif model_name in ['VGG16', 'ResNet50']:
        return load_cnn(model_name)
    elif model_name == 'SVM':
        return load_svm()
    return None


def get_available_models(data_type: str) -> list[str]:
    """Get models compatible with data type that have weights."""
    compatible = DATA_TYPE_CONFIG[data_type]['models']
    return [m for m in compatible if MODEL_PATHS[m].exists()]


# =============================================================================
# Visualization
# =============================================================================

def create_iq_plot(iq_data: np.ndarray) -> go.Figure:
    """Create IQ signal visualization. Input: (2, n_samples)"""
    n_samples = iq_data.shape[1]

    x = np.arange(n_samples)
    i_data, q_data = iq_data[0], iq_data[1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=i_data, mode='lines', name='I',
                             line=dict(color='#e74c3c', width=1)))
    fig.add_trace(go.Scatter(x=x, y=q_data, mode='lines', name='Q',
                             line=dict(color='#2ecc71', width=1)))
    fig.update_layout(
        title='IQ Signal - 20ms Segment',
        xaxis_title='Sample', yaxis_title='Amplitude (normalized)',
        template='plotly_dark', height=350,
        margin=dict(l=50, r=20, t=30, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def create_psd_plot(psd_data: np.ndarray) -> go.Figure:
    """Create PSD visualization. Input: (n_bins,)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=psd_data, mode='lines', name='PSD',
                             line=dict(color='#3498db', width=1.5)))
    fig.update_layout(
        title='Power Spectral Density',
        xaxis_title='Frequency Bin', yaxis_title='PSD (normalized linear)',
        template='plotly_dark', height=350,
        margin=dict(l=50, r=20, t=30, b=50)
    )
    return fig


def create_spectrogram_plot(spec_data: np.ndarray) -> go.Figure:
    """Create spectrogram visualization. Input: (H, W, 3) or (H, W)"""
    if spec_data.ndim == 3:
        display_data = np.mean(spec_data, axis=-1) if spec_data.shape[-1] == 3 else spec_data[:, :, 0]
    else:
        display_data = spec_data

    fig = go.Figure(data=go.Heatmap(z=display_data, colorscale='Viridis', colorbar=dict(title='Intensity (normalized)')))
    fig.update_layout(
        title='Spectrogram - 20ms Segment',
        xaxis_title='Time (ms)', yaxis_title='Frequency offset (MHz)',
        template='plotly_dark', height=350,
        margin=dict(l=50, r=20, t=30, b=50)
    )
    return fig


def create_proba_chart(probabilities: np.ndarray, classes: list,
                       true_idx: int, color: str) -> go.Figure:
    """Create probability bar chart with true class highlighted."""
    colors = [color if i != true_idx else '#2ecc71' for i in range(len(classes))]
    fig = go.Figure(data=[
        go.Bar(
            x=list(classes), y=probabilities * 100,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probabilities * 100],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title='Class Probabilities',
        xaxis_title="Class", yaxis_title="Probability (%)",
        yaxis_range=[0, 100], template='plotly_dark', height=300,
        margin=dict(l=50, r=20, t=30, b=50)
    )
    return fig


VIZ_FUNCS = {
    'iq': create_iq_plot,
    'psd': create_psd_plot,
    'spectrogram': create_spectrogram_plot
}


# =============================================================================
# View Class
# =============================================================================

class InferenceView:
    """Inference view controller."""

    def render(self):
        """Main render method."""
        # Sidebar - Sample Selection
        with st.sidebar:
            st.header('Sample Selection')

            # 1. Interference
            interferences = list_available_interferences()
            if not interferences:
                st.error(f'No test samples found in {TEST_SAMPLES_DIR}')
                return
            selected_interference = st.selectbox('Interference', interferences)

            # 2. Data Type
            types = list_available_types(selected_interference)
            if not types:
                st.error('No data types found')
                return
            type_names = {t: DATA_TYPE_CONFIG[t]['name'] for t in types}
            selected_type = st.selectbox(
                'Data Type', types,
                format_func=lambda x: f"{type_names[x]} ({x})"
            )

            # 3. Drone
            drones = list_available_drones(selected_interference, selected_type)
            if not drones:
                st.error('No drones found')
                return
            selected_drone = st.selectbox(
                'Drone', drones,
                format_func=lambda x: f"{x} - {DRONE_INFO.get(x, {}).get('name', '?')}"
            )

            # Load sample file
            sample_path = get_sample_filepath(selected_interference, selected_type, selected_drone)
            if not sample_path or not sample_path.exists():
                st.error(f'Sample file not found')
                return

            data = load_sample_file(str(sample_path))
            if data is None:
                return
            n_samples = len(data['y'])

            # 4. Sample Index
            st.caption(f"Available samples: **{n_samples}**")
            sample_idx = st.number_input('Sample Index', min_value=0,
                                         max_value=n_samples - 1, value=0)

            st.divider()

            # 5. Model Selection
            st.header('Model')
            available_models = get_available_models(selected_type)
            compatible_models = DATA_TYPE_CONFIG[selected_type]['models']

            if not available_models:
                st.warning(f"No models available for {selected_type}")
                st.caption(f"Expected: {', '.join(compatible_models)}")
                selected_model = None
                run_inference = False
            else:
                selected_model = st.selectbox('Select Model', available_models)
                st.success(f'{selected_model} ready ({DEVICE})')
                run_inference = st.button('Run Inference', type='primary', width='stretch')

        # Main Panel
        sample = data['X'][sample_idx]
        true_label_idx = int(data['y'][sample_idx])
        true_drone = DRONE_CLASSES[true_label_idx]

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader(DATA_TYPE_CONFIG[selected_type]['name'])
            viz_func = VIZ_FUNCS.get(selected_type)
            if viz_func:
                st.plotly_chart(viz_func(sample), width='stretch')
            else:
                st.error(f'No visualization for {selected_type}')

        with col2:
            st.subheader('Sample Info')
            drone_info = DRONE_INFO.get(true_drone, {})
            st.metric('True Class', f"{true_drone} ({drone_info.get('name', '')})")

            c1, c2 = st.columns(2)
            with c1:
                st.metric('Interference', selected_interference)
            with c2:
                st.metric('Protocol', drone_info.get('protocol', 'N/A'))

            st.caption(f"Shape: `{sample.shape}`")
            st.caption(f"Type: `{selected_type}`")
            st.caption(f"File: `{sample_path.name}`")

        # Inference Results
        if run_inference and selected_model:
            st.divider()
            st.subheader('Inference Results')

            model = load_model(selected_model)
            if model is None:
                st.error(f'Failed to load {selected_model}')
                return

            # Run inference
            start = time.perf_counter()
            proba = model.predict_proba(sample)
            elapsed = (time.perf_counter() - start) * 1000

            pred_idx = int(np.argmax(proba))
            pred_drone = DRONE_CLASSES[pred_idx]
            confidence = proba[pred_idx] * 100
            is_correct = pred_drone == true_drone

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            pred_info = DRONE_INFO.get(pred_drone, {})
            with c1:
                st.metric('Prediction', f"{pred_drone} ({pred_info.get('name', '')})")
            with c2:
                st.metric('Confidence', f"{confidence:.1f}%")
            with c3:
                st.metric('Inference Time', f"{elapsed:.2f} ms")
            with c4:
                if is_correct:
                    st.success('CORRECT')
                else:
                    st.error(f'WRONG (true: {true_drone})')

            # Probability chart
            model_color = COLORS.get(selected_model, '#888888')
            st.plotly_chart(
                create_proba_chart(proba, DRONE_CLASSES, true_label_idx, model_color),
                width='stretch'
            )
