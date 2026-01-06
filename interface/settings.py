"""
Configuration settings for DroneDetect inference application.
Equivalent to Django settings.py
"""
from pathlib import Path
import torch
import numpy as np
from dronedetect.config import MODEL_INPUT_SHAPES as _MODEL_INPUT_SHAPES

# Project paths
BASE_DIR = Path(__file__).parent.parent
INTERFACE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'notebooks/data/features'
MODELS_DIR = BASE_DIR / 'models'
TEST_SAMPLES_DIR = INTERFACE_DIR / 'media/test_samples'

# Hardware configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Drone classes (7 classes from DroneDetect V2 dataset)
DRONE_CLASSES = np.array(['AIR', 'DIS', 'INS', 'MA1', 'MAV', 'MIN', 'PHA'])
NUM_CLASSES = len(DRONE_CLASSES)

# Drone metadata from DroneDetect V2 dataset
DRONE_INFO = {
    'AIR': {
        'name': 'DJI Air 2S',
        'protocol': 'OcuSync 3.0',
        'weight': 595,
        'max_speed': 68.4,
        'eirp': '26 dBm',
    },
    'DIS': {
        'name': 'Parrot Disco',
        'protocol': 'Wi-Fi (SkyController 2)',
        'weight': 750,
        'max_speed': 80,
        'eirp': '19 dBm',
    },
    'INS': {
        'name': 'DJI Inspire 2',
        'protocol': 'Lightbridge 2.0',
        'weight': 3440,
        'max_speed': 94,
        'eirp': '20 dBm',
    },
    'MA1': {
        'name': 'DJI Mavic 2 Pro',
        'protocol': 'OcuSync 2.0',
        'weight': 907,
        'max_speed': 72,
        'eirp': '25.5 dBm',
    },
    'MAV': {
        'name': 'DJI Mavic Pro',
        'protocol': 'OcuSync 1.0',
        'weight': 734,
        'max_speed': 65,
        'eirp': '26 dBm',
    },
    'MIN': {
        'name': 'DJI Mavic Mini',
        'protocol': 'Wi-Fi',
        'weight': 249,
        'max_speed': 46.8,
        'eirp': '19 dBm',
    },
    'PHA': {
        'name': 'DJI Phantom 4',
        'protocol': 'Lightbridge 2.0',
        'weight': 1380,
        'max_speed': 72,
        'eirp': '20 dBm',
    },
}

# Interference types
INTERFERENCE_TYPES = ['CLEAN', 'BOTH']

# Data type configuration
DATA_TYPE_CONFIG = {
    'psd': {
        'name': 'PSD (Power Spectral Density)',
        'models': ['SVM'],
        'description': 'Frequency domain features for SVM classifier',
        'viz_type': 'line',
        'x_label': 'Frequency Bin',
        'y_label': 'Power (dB)',
        'file_path': DATA_DIR / 'psd_features.npz',
        'sample_prefix': 'psd',
        'sample_suffix': '20',
    },
    'spectrogram': {
        'name': 'Spectrogram',
        'models': ['VGG16', 'ResNet50'],
        'description': 'Time-frequency representation for CNN classifiers',
        'viz_type': 'heatmap',
        'x_label': 'Time',
        'y_label': 'Frequency',
        'file_path': DATA_DIR / 'spectrogram_features.npz',
        'sample_prefix': 'spectrogram',
        'sample_suffix': '224x224x3',
    },
    'iq': {
        'name': 'IQ Signal (In-phase/Quadrature)',
        'models': ['RFUAVNet'],
        'description': 'Raw complex signal for RF-UAV-Net',
        'viz_type': 'dual_line',
        'x_label': 'Sample',
        'y_label': 'Amplitude',
        'file_path': DATA_DIR / 'iq_features.npz',
        'sample_prefix': 'iq',
        'sample_suffix': '20',
    }
}

MODEL_TO_DATA_TYPE = {
    'SVM': 'psd',
    'VGG16': 'spectrogram',
    'ResNet50': 'spectrogram',
    'RFUAVNet': 'iq'
}

# Model input sizes (with batch dimension for torch)
MODEL_INPUT_SIZES = {
    name: (1, *shape) for name, shape in _MODEL_INPUT_SHAPES.items()
}

# Model paths
MODEL_PATHS = {
    'SVM': MODELS_DIR / 'svm_psd_drone.pkl',
    'VGG16': MODELS_DIR / 'vgg16_cnn.pth',
    'ResNet50': MODELS_DIR / 'resnet50_cnn.pth',
    'RFUAVNet': MODELS_DIR / 'rfuavnet_iq.pth',
}

# UI configuration
COLORS = {
    'SVM': '#3498db',
    'VGG16': '#e74c3c',
    'ResNet50': '#2ecc71',
    'RFUAVNet': '#9b59b6'
}

# Streamlit configuration
PAGE_CONFIG = {
    'page_title': 'DroneDetect V2',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}


def get_test_sample_path(data_type: str, interference: str, drone_class: str) -> Path:
    """Get path to a test sample file."""
    config = DATA_TYPE_CONFIG[data_type]
    prefix = config['sample_prefix']
    suffix = config['sample_suffix']
    filename = f"{prefix}_{interference}_{drone_class}_{suffix}.npz"
    return TEST_SAMPLES_DIR / interference / filename
