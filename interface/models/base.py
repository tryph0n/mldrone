"""
Base model interface for DroneDetect models.
Enforces consistent API across SVM and PyTorch models.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Base class for all DroneDetect models."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> int:
        """Return predicted class index."""
        pass

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        pass

    @abstractmethod
    def get_required_data_type(self) -> str:
        """Return required data type: 'psd', 'spectrogram', or 'iq'."""
        pass
