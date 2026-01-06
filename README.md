# DroneDetect V2 - RF Classification Pipeline

Machine learning pipeline for drone RF signal classification using the DroneDetect V2 dataset.


## References

| Paper | Contribution |
|-------|--------------|
| [Swinney & Woods, 2021](https://doi.org/10.3390/aerospace8070179) | DroneDetect dataset, PSD vs spectrogram comparison |
| [RF-UAVNet (IEEE, 2022)](https://ieeexplore.ieee.org/document/9768809) | 1D CNN architecture for drone classification (raw IQ signals) |
| [ Kiliç et al., 2021](https://doi.org/10.1016/j.jestch.2021.06.008) | RF signal classification with machine learning |
| [ Swinney et al., 2021](https://doi.org/10.3390/aerospace8030079) | Flying mode classification via ResNet50 |


## Dataset

[**DroneDetect V2** (Swinney & Woods, 2021)](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine)
- 60 MHz sampling rate
- 2.43 GHz center frequency
- 7 drone types: DJI Air 2S, Parrot Disco, DJI Inspire 2, DJI Mavic Mini, DJI Mavic Pro 1/2, DJI Phantom 4
- 3 states: ON, Hovering (HO), Flying (FY)
- 4 interference conditions: CLEAN, Bluetooth, WiFi, Both

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install with PyTorch CPU
git clone <repo-url> && cd mldrone
uv sync --extra torch-cpu

# OR with PyTorch GPU (CUDA 12.1)
uv sync --extra torch-gpu

# Optional: add visualization tools (torchview, torchinfo)
uv sync --extra torch-cpu --extra viz

# Extract dataset
unzip /path/to/DroneDetect_V2.zip -d ./data/

# Test installation
uv run python -c "import torch; print(torch.__version__)"
```

## Configuration

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DRONEDETECT_DATA_DIR` | `./data/raw` | Raw dataset directory |
| `DRONEDETECT_FEATURES_DIR` | `./data/features` | Extracted features |
| `DRONEDETECT_MODELS_DIR` | `./models` | Trained models |
| `DRONEDETECT_METADATA_CACHE` | `./data/metadata_cache.parquet` | Metadata cache |

## Usage

### 1. Explore Dataset

Open notebooks in your IDE (VSCode, PyCharm, etc.) with the `.venv` Python interpreter. (check out `figures` folder and related pdf for output)

### 2. Extract Features

Run preprocessing notebooks to extract:
- PSD features (for SVM)
- Spectrograms (224x224 for CNNs)
- Downsampled IQ (2x10000 for RF-UAV-Net)

Features saved to `./data/features/*.npz`

### 3. Train Models

- `021_training_svm.ipynb`: SVM on PSD features
- `022_training_cnn.ipynb`: VGG16 + ResNet50 on spectrograms
- `023_training_rfuavnet.ipynb`: 1D CNN on raw IQ

Models saved to `./models/*.pth` or `*.pkl`

### 4. Run Inference

```bash
uv run streamlit run interface/app.py
```
See inference page


## Models

| Model | Input |
|-------|-------|
| SVM (RBF) | PSD |
| VGG16-FC | Spectrogram |
| ResNet50-FC | Spectrogram |
| RF-UAV-Net | Raw IQ |


## License

MIT