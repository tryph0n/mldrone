"""Feature extraction functions for RF signals."""

import numpy as np
from scipy import signal, ndimage
from typing import Tuple
from .config import FS, DEFAULT_NFFT, DEFAULT_NOVERLAP, IMG_SIZE


def compute_psd(
    segment: np.ndarray,
    fs: float = FS,
    nfft: int = DEFAULT_NFFT,
    window: str = 'hamming'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-sided Power Spectral Density for complex signals.
    Returns BASEBAND frequency offsets (relative to 0 Hz center).

    Reference: REFERENTIEL Section 1.2 - PSD remains in LINEAR scale.
    dB conversion is ONLY applied to spectrograms, NOT to PSD features.

    Args:
        segment: Complex IQ signal (baseband representation)
        fs: Sampling frequency
        nfft: FFT length (default: 1024 per reference)
        window: Window function ('hamming' per reference)

    Returns:
        freqs: Frequency OFFSETS in Hz, range [-fs/2, +fs/2] (e.g., [-30e6, +30e6])
               These are relative to baseband center (0 Hz), NOT absolute RF.
               To convert: f_absolute = freqs + CENTER_FREQ (e.g., freqs + 2.4375e9)
        psd: Power spectral density in LINEAR scale (NOT dB)

    Example:
        freqs, psd = compute_psd(segment)
        # freqs[512] ≈ +2.6e6 Hz (offset)
        # Absolute RF: 2.6e6 + 2.4375e9 = 2440.1 MHz
    """
    from scipy.signal import welch

    # For complex signals, return_onesided must be False
    freqs, psd = welch(segment, fs=fs, nperseg=nfft,
                       window=window, return_onesided=False)

    # Shift to center spectrum: [-fs/2, +fs/2]
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    # NO dB conversion - keep linear scale (reference: Section 1.2.2)
    # Per-sample normalization (div by max) is applied in PSDStep, not here

    return freqs, psd


def compute_spectrogram(
    segment: np.ndarray,
    fs: float = FS,
    nfft: int = DEFAULT_NFFT,
    noverlap: int = DEFAULT_NOVERLAP,
    target_size: Tuple[int, int] = IMG_SIZE
) -> np.ndarray:
    """
    Compute spectrogram via STFT and resize to target_size.

    Reference: REFERENTIEL Section 2.2 - Spectrogram generation for VGG16/ResNet50.
    - STFT with n_fft=1024, hanning window, noverlap=120
    - Conversion to dB with NEGATIVE sign: -10*log10()
    - Viridis colormap to generate RGB image (3 channels)
    - Bilinear interpolation to 224x224
    - Normalization to [0, 1] per image

    Returns: RGB array of shape (target_size[0], target_size[1], 3) in [0, 1].
    """
    from matplotlib import mlab, cm

    # Use scipy.signal.spectrogram for complex signals
    # Reference: REFERENTIEL Section 2.2.1, lines 380-382
    from scipy.signal import spectrogram as scipy_spectrogram

    # Compute spectrogram (two-sided for complex signals)
    freqs, times, spec = scipy_spectrogram(
        segment, fs=fs, nperseg=nfft, noverlap=noverlap,
        window='hann', return_onesided=False, mode='magnitude'
    )

    # Extract positive frequencies only
    n_positive = nfft // 2 + 1
    spec = spec[:n_positive, :]

    # Resize to target dimensions (bilinear interpolation, reference: Section 2.2.3)
    zoom_factors = (target_size[0] / spec.shape[0], target_size[1] / spec.shape[1])
    spec_resized = ndimage.zoom(spec, zoom_factors, order=1)

    # Convert to dB with NEGATIVE sign (reference: Section 1.2.2, line 154)
    spec_db = -10 * np.log10(spec_resized + 1e-12)

    # Normalize to [0, 1] with division-by-zero protection
    spec_min = spec_db.min()
    spec_max = spec_db.max()
    range_db = spec_max - spec_min

    if range_db < 1e-10:  # Flat spectrum (all same value)
        spec_norm = np.ones_like(spec_db) * 0.5  # Return mid-gray
    else:
        spec_norm = (spec_db - spec_min) / range_db

    # Apply viridis colormap to generate RGB image (reference: Section 2.2.5)
    viridis = cm.get_cmap('viridis')
    spec_rgb = viridis(spec_norm)[:, :, :3]  # Discard alpha channel, keep RGB only

    return spec_rgb.astype(np.float32)  # Shape: (height, width, 3), explicit float32


def interpolate_2d(arr: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize 2D array to output_size via bilinear interpolation."""
    zoom_factors = (output_size[0] / arr.shape[0], output_size[1] / arr.shape[1])
    return ndimage.zoom(arr, zoom_factors, order=1)
