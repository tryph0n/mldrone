"""Configuration settings for DroneDetect Streamlit application."""

from pathlib import Path
import torch
import numpy as np
from dronedetect.config import MODEL_INPUT_SHAPES as _MODEL_INPUT_SHAPES

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "data" / "models"
TEST_SAMPLES_DIR = BASE_DIR / "data" / "test_samples"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DRONE_CLASSES = np.array(["AIR", "DIS", "INS", "MA1", "MAV", "MIN", "PHA"])
NUM_CLASSES = len(DRONE_CLASSES)

DRONE_INFO = {
    "AIR": {
        "name": "DJI Air 2S",
        "protocol": "OcuSync 3.0",
        "weight": 595,
        "max_speed": 68.4,
        "eirp": "26 dBm",
    },
    "DIS": {
        "name": "Parrot Disco",
        "protocol": "Wi-Fi (SkyController 2)",
        "weight": 750,
        "max_speed": 80,
        "eirp": "19 dBm",
    },
    "INS": {
        "name": "DJI Inspire 2",
        "protocol": "Lightbridge 2.0",
        "weight": 3440,
        "max_speed": 94,
        "eirp": "20 dBm",
    },
    "MA1": {
        "name": "DJI Mavic 2 Pro",
        "protocol": "OcuSync 2.0",
        "weight": 907,
        "max_speed": 72,
        "eirp": "25.5 dBm",
    },
    "MAV": {
        "name": "DJI Mavic Pro",
        "protocol": "OcuSync 1.0",
        "weight": 734,
        "max_speed": 65,
        "eirp": "26 dBm",
    },
    "MIN": {
        "name": "DJI Mavic Mini",
        "protocol": "Wi-Fi",
        "weight": 249,
        "max_speed": 46.8,
        "eirp": "19 dBm",
    },
    "PHA": {
        "name": "DJI Phantom 4",
        "protocol": "Lightbridge 2.0",
        "weight": 1380,
        "max_speed": 72,
        "eirp": "20 dBm",
    },
}
INTERFERENCE_TYPES = ["CLEAN", "WIFI", "BLUE", "BOTH"]

DATA_TYPE_CONFIG = {
    "psd": {
        "name": "PSD (Power Spectral Density)",
        "models": ["SVM"],
        "description": "Frequency domain features for SVM classifier",
    },
    "spectrogram": {
        "name": "Spectrogram",
        "models": ["VGG16", "ResNet50"],
        "description": "Time-frequency representation for CNN classifiers",
    },
    "iq": {
        "name": "IQ Signal (In-phase/Quadrature)",
        "models": ["RFUAVNet"],
        "description": "Raw complex signal for RF-UAV-Net",
    },
}
MODEL_INPUT_SIZES = {name: (1, *shape) for name, shape in _MODEL_INPUT_SHAPES.items()}

MODEL_PATHS = {
    "SVM": MODELS_DIR / "svm_psd_drone.pkl",
    "VGG16": MODELS_DIR / "vgg16_cnn.pth",
    "ResNet50": MODELS_DIR / "resnet50_cnn.pth",
    "RFUAVNet": MODELS_DIR / "rfuavnet_iq.pth",
}

COLORS = {
    "SVM": "#3498db",
    "VGG16": "#e74c3c",
    "ResNet50": "#2ecc71",
    "RFUAVNet": "#9b59b6",
}

PAGE_CONFIG = {
    "page_title": "DroneDetect V2",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}
