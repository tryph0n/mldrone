"""Preprocessing utilities for RF signal data."""

import numpy as np

from .config import FS, DEFAULT_SEGMENT_MS, DEFAULT_IQ_DOWNSAMPLE


def normalize(signal: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit variance normalization (Z-score).

    Used for PSD and Spectrogram features.
    """
    return (signal - np.mean(signal)) / np.sqrt(np.var(signal))


def normalize_minmax(signal: np.ndarray) -> np.ndarray:
    """
    Min-max normalization [0, 1] per I/Q channel independently (per-file).

    Used ONLY for RF-UAVNet IQ features to match RFClassification approach
    (while avoiding their data leakage by normalizing per-file).

    RFClassification uses global min-max BEFORE split (data leakage).
    We use per-file min-max BEFORE segmentation (leakage-free).

    Args:
        signal: Complex IQ signal for ONE file

    Returns:
        Normalized signal with I and Q channels scaled to [0, 1]

    References:
        - IQTLabs RFClassification: github.com/IQTLabs/RFClassification
        - arXiv:2308.11833 - Robust RF Data Normalization
    """
    segment_I = signal.real
    segment_Q = signal.imag

    # Min-max [0, 1] per channel
    I_range = segment_I.max() - segment_I.min()
    Q_range = segment_Q.max() - segment_Q.min()

    # Avoid division by zero
    I_norm = (segment_I - segment_I.min()) / (I_range if I_range > 1e-12 else 1.0)
    Q_norm = (segment_Q - segment_Q.min()) / (Q_range if Q_range > 1e-12 else 1.0)

    return I_norm + 1j * Q_norm


def segment_signal(
    signal: np.ndarray, segment_ms: float = DEFAULT_SEGMENT_MS, fs: float = FS
) -> list[np.ndarray]:
    """
    Segment signal into fixed-length windows.

    With fs=60MHz and segment_ms=20, each segment = 1,200,000 samples.
    Returns list of segments (drops incomplete final segment).
    """
    samples_per_segment = int(segment_ms / 1e3 * fs)
    n_segments = len(signal) // samples_per_segment
    n_keep = n_segments * samples_per_segment

    return np.array_split(signal[:n_keep], n_segments)


def downsample_iq(
    segment: np.ndarray, target_samples: int = DEFAULT_IQ_DOWNSAMPLE
) -> np.ndarray:
    """
    Downsample IQ segment via linear interpolation.

    Returns: array of shape (2, target_samples) with [real, imag] channels.
    """
    t_old = np.arange(len(segment))
    t_new = np.linspace(0, len(segment) - 1, num=target_samples)

    # numpy.interp is more efficient than scipy.interpolate.interp1d
    real_down = np.interp(t_new, t_old, segment.real)
    imag_down = np.interp(t_new, t_old, segment.imag)

    return np.stack([real_down, imag_down], axis=0)
