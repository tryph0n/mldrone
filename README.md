# DroneDetect V2 - RF Classification Pipeline

Machine learning pipeline for drone RF signal classification using the DroneDetect V2 dataset.

## Key Contribution

The original [IQTLabs/RFClassification](https://github.com/IQTLabs/RFClassification) repository uses a **segment-level random split**: spectrogram segments extracted from the same recording file can appear in both training and test sets. Because consecutive segments from a single recording are highly correlated, this constitutes **data leakage** and yields artificially inflated accuracy (~99.8%).

This project implements a **file-level split** (70/15/15 train/val/test, drone-only stratification) that guarantees all segments from one recording stay in the same partition. Under this honest evaluation protocol, model performance drops significantly:

| Model | Input | Accuracy |
|-------|-------|----------|
| VGG16-FC | Spectrogram | 88.76% |
| SVM (RBF) | PSD | 83.46% |
| ResNet50-FC | Spectrogram | 78.80% |
| RF-UAVNet | Raw IQ | 62.98% |

These results reflect genuine generalization to unseen recordings rather than memorization of correlated segments.


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
| `DRONEDETECT_MODELS_DIR` | `./data/models` | Trained models |
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

Models saved to `./data/models/*.pth` or `*.pkl`

### 4. Run Inference

```bash
uv run streamlit run interface/app.py
```
See inference page

## CLI Commands

All CLI commands are run via `uv run python -m <module>`.

### Feature Extraction Pipeline

Downloads raw .dat files from S3, extracts PSD, spectrogram, and IQ features, then saves results to disk. Supports resume via checkpointing and optional S3 upload.

```bash
uv run python -m dronedetect.pipeline [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--features` | `all` | Features to extract: `all`, `psd`, `spectrogram`, `iq` (comma-separated) |
| `--concurrency` | `2` | Number of parallel workers for feature extraction |
| `--output-dir` | `DRONEDETECT_FEATURES_DIR` | Output directory for feature files |
| `--upload` | off | Upload results to S3 after extraction |
| `--tmp-dir` | system temp | Temporary directory for .dat downloads |
| `--max-files` | none | Process only the first N files (for testing) |
| `--verbose` | off | Enable debug logging |

```bash
# Extract all features with default settings
uv run python -m dronedetect.pipeline

# Extract only PSD and spectrogram with 4 workers
uv run python -m dronedetect.pipeline --features psd,spectrogram --concurrency 4

# Test run on 5 files with upload
uv run python -m dronedetect.pipeline --max-files 5 --upload --verbose
```

### Upload Artifacts to S3

Uploads project artifacts (features, models, split indices) to the Scaleway S3 bucket. At least one artifact flag is required.

```bash
uv run python -m dronedetect.upload [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--features` | Upload feature files from `data/features` |
| `--models` | Upload model files from `data/models` |
| `--split` | Upload split indices from `data/split_indices.npz` |
| `--all` | Upload all three (equivalent to `--features --models --split`) |

```bash
# Upload everything
uv run python -m dronedetect.upload --all

# Upload only models
uv run python -m dronedetect.upload --models
```

### Export Test Samples

Exports one representative file (100 segments) per (drone, condition) combination from the test set, organized by interference condition. Used to populate the Streamlit inference interface.

```bash
uv run python -m dronedetect.export_samples [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--features-dir` | `DRONEDETECT_FEATURES_DIR` | Directory containing pipeline feature files |
| `--split-path` | `<features-dir>/../split_indices.npz` | Path to the split indices file |
| `--output-dir` | `DRONEDETECT_TEST_SAMPLES_DIR` | Output directory for test samples |
| `--verbose` | off | Enable verbose logging |

```bash
# Export with default paths
uv run python -m dronedetect.export_samples

# Export with custom directories
uv run python -m dronedetect.export_samples \
  --features-dir ./data/features \
  --output-dir ./data/test_samples \
  --verbose
```

## Models

| Model | Input |
|-------|-------|
| SVM (RBF) | PSD |
| VGG16-FC | Spectrogram |
| ResNet50-FC | Spectrogram |
| RF-UAV-Net | Raw IQ |


## Credits

This project builds upon and extends the following work:

- **[IQTLabs/RFClassification](https://github.com/IQTLabs/RFClassification)** -- Original RF drone classification pipeline (Apache-2.0 license). Our project fixes the data leakage issue in their evaluation protocol and reimplements the full pipeline with file-level splitting.

- **DroneDetect V2 Dataset** -- Swinney, C.J. & Woods, J.C. (2021). *DroneDetect Dataset: A Radio Frequency Dataset of Unmanned Aerial System (UAS) Signals for Machine Learning Detection and Classification*. IEEE DataPort. DOI: [10.21227/6w92-0x42](https://doi.org/10.21227/6w92-0x42)

- **RF-UAVNet Architecture** -- Huynh-The, T. et al. (2022). *Lightweight RF-UAVNet for Drone Surveillance Systems*. IEEE Access, vol. 10, pp. 92390-92400. DOI: [10.1109/ACCESS.2022.3203214](https://doi.org/10.1109/ACCESS.2022.3203214)


## License

MIT