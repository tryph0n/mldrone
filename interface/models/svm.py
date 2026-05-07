"""
SVM model wrapper for consistent interface.
"""

import numpy as np

from .base import BaseModel


class SVMAdapter(BaseModel):
    """Adapter for sklearn SVM to match BaseModel interface."""

    def __init__(self, svm_dict_or_model):
        if isinstance(svm_dict_or_model, dict):
            self.model = svm_dict_or_model["model"]
            self.classes = svm_dict_or_model.get("classes", None)
        else:
            self.model = svm_dict_or_model
            self.classes = None

    def predict(self, x: np.ndarray) -> int:
        x = x.reshape(1, -1)
        return int(self.model.predict(x)[0])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(1, -1)
        if (
            hasattr(self.model, "predict_proba")
            and hasattr(self.model, "probability")
            and self.model.probability
        ):
            return self.model.predict_proba(x)[0]
        else:
            decision = self.model.decision_function(x)[0]
            exp_decision = np.exp(decision - np.max(decision))
            return exp_decision / exp_decision.sum()

    def get_required_data_type(self) -> str:
        return "psd"
