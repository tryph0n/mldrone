"""Data loading utilities for DroneDetect V2 dataset."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from .config import *


def load_raw_iq(file_path: str | Path) -> np.ndarray:
    """
    Load a DroneDetect .dat file as complex IQ samples.

    Args:
        file_path: Path to .dat file

    Returns:
        Complex64 array of shape (120_000_000,)
    """
    data = np.fromfile(file_path, dtype=np.float32, count=RAW_SAMPLE_COUNT)
    return data.astype(np.float32).view(np.complex64)


def parse_filename(filename: str) -> dict:
    """
    Parse DroneDetect filename to extract metadata.

    Example: 'MA1_0110_02.dat' ->
        {drone_code: 'MA1', wifi: False, bluetooth: True, state: 'FY', index: 2}
    """
    parts = filename.replace('.dat', '').split('_')
    drone_code = parts[0]
    code = parts[1]
    index = int(parts[2])

    return {
        "drone_code": drone_code,
        "drone_folder": DRONE_CODE_TO_FOLDER.get(drone_code, drone_code),
        "wifi": code[0] == '1',
        "bluetooth": code[1] == '1',
        "interference": INTERFERENCE_MAP_INV.get(code[:2], "UNKNOWN"),
        "state": STATE_MAP_INV.get(code[2:4], "UNKNOWN"),
        "index": index
    }


def get_dataset_metadata(root_dir: str | Path) -> pd.DataFrame:
    """
    Scan dataset directory and return metadata DataFrame.

    Returns DataFrame with columns:
        file_path, drone_code, drone_folder, interference, state, index, wifi, bluetooth
    """
    root = Path(root_dir)
    records = []

    for interference_dir in root.iterdir():
        if not interference_dir.is_dir():
            continue
        interference = interference_dir.name

        for drone_state_dir in interference_dir.iterdir():
            if not drone_state_dir.is_dir():
                continue

            for dat_file in drone_state_dir.glob("*.dat"):
                meta = parse_filename(dat_file.name)
                meta["file_path"] = str(dat_file)
                meta["interference_folder"] = interference
                records.append(meta)

    return pd.DataFrame(records)


def get_cached_metadata(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load dataset metadata from cache or generate and cache it.

    Args:
        force_refresh: If True, regenerate cache even if it exists

    Returns:
        DataFrame with dataset metadata
    """
    cache_path = METADATA_CACHE

    if not force_refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    df = get_dataset_metadata(DATA_DIR)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df
