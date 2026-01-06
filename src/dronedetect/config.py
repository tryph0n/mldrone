"""Configuration parameters for DroneDetect V2 dataset and processing."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Dataset parameters
FS = 60e6  # 60 MHz sampling rate
BANDWIDTH = 28e6  # 28 MHz
CENTER_FREQ = 2.4375e9  # 2.4375 GHz
RAW_SAMPLE_COUNT = 240_000_000  # float32 values per file
COMPLEX_SAMPLE_COUNT = 120_000_000  # complex64 samples per file

# Preprocessing
DEFAULT_SEGMENT_MS = 20  # milliseconds
DEFAULT_NFFT = 1024  # Reference: REFERENTIEL Section 1.2.1 (nperseg=1024)
DEFAULT_NOVERLAP = 120
DEFAULT_IQ_DOWNSAMPLE = 10_000

# Image output
IMG_SIZE = (224, 224)

# Model input shapes (batch size excluded)
MODEL_INPUT_SHAPES = {
    'SVM': (DEFAULT_NFFT,),  # PSD features
    'VGG16': (*IMG_SIZE, 3),  # RGB spectrogram (H, W, C)
    'ResNet50': (*IMG_SIZE, 3),  # RGB spectrogram (H, W, C)
    'RFUAVNet': (2, DEFAULT_IQ_DOWNSAMPLE),  # IQ channels (I+Q, samples)
}

# Drone code mapping (folder name -> file code)
DRONE_FOLDER_TO_CODE = {
    "AIR": "AIR", # DJI Air 2S
    "DIS": "DIS", # Parrot Disco
    "INS": "INS", # DJI Inspire 2
    "MIN": "MIN", # DJI Mavic Mini 
    "MP1": "MA1", # DJI Mavic Pro
    "MP2": "MAV", # DJI Mavic Pro 2
    "PHA": "PHA" # DJI Phantom 4
}
DRONE_CODE_TO_FOLDER = {v: k for k, v in DRONE_FOLDER_TO_CODE.items()}

# Interference mapping
INTERFERENCE_MAP = {"CLEAN": "00", "BLUE": "01", "WIFI": "10", "BOTH": "11"}
INTERFERENCE_MAP_INV = {v: k for k, v in INTERFERENCE_MAP.items()}

# State mapping (positions 2-3 of code)
STATE_MAP = {"ON": "00", "HO": "01", "FY": "10"}
STATE_MAP_INV = {v: k for k, v in STATE_MAP.items()}

# Paths
DATA_DIR = Path(os.environ.get("DRONEDETECT_DATA_DIR", "./data/raw"))
FEATURES_DIR = Path(os.environ.get("DRONEDETECT_FEATURES_DIR", "./data/features"))
MODELS_DIR = Path(os.environ.get("DRONEDETECT_MODELS_DIR", "./models"))
METADATA_CACHE = Path(os.environ.get("DRONEDETECT_METADATA_CACHE", "./data/metadata_cache.parquet"))
