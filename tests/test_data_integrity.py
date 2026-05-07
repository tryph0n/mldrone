"""Data integrity tests for the DroneDetect V2 feature pipeline.

Validates that PSD, spectrogram, and IQ features are consistent with each
other and that the train/val/test split is leakage-free.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from dronedetect.splitting import create_stratified_split, load_split

logger = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
SPLIT_PATH = Path(__file__).parent.parent / "data" / "split_indices.npz"

PSD_PATH = FEATURES_DIR / "psd_features.npz"
SPECTRO_PATH = FEATURES_DIR / "spectrogram_features.npy"
SPECTRO_META_PATH = FEATURES_DIR / "spectrogram_meta.npz"
IQ_PATH = FEATURES_DIR / "iq_features.npz"

HAS_PSD = PSD_PATH.exists()
HAS_SPECTRO = SPECTRO_PATH.exists() and SPECTRO_META_PATH.exists()
HAS_IQ = IQ_PATH.exists()
HAS_SPLIT = SPLIT_PATH.exists()
HAS_ALL_FEATURES = HAS_PSD and HAS_SPECTRO and HAS_IQ


def _load_psd():
    data = np.load(PSD_PATH)
    return {
        "X": data["X"],
        "y_drone": data["y_drone"],
        "file_ids": data["file_ids"],
    }


def _load_spectro():
    X = np.load(SPECTRO_PATH, mmap_mode="r")
    meta = np.load(SPECTRO_META_PATH)
    return {
        "X": X,
        "y_drone": meta["y_drone"],
        "file_ids": meta["file_ids"],
    }


def _load_iq():
    data = np.load(IQ_PATH)
    return {
        "X": data["X"],
        "y_drone": data["y_drone"],
        "file_ids": data["file_ids"],
    }


@pytest.mark.skipif(not HAS_PSD, reason="PSD features not found")
class TestPsdConsistency:
    def test_sample_count_matches(self):
        d = _load_psd()
        n = d["X"].shape[0]
        assert len(d["y_drone"]) == n, f"y_drone length {len(d['y_drone'])} != X rows {n}"
        assert len(d["file_ids"]) == n, f"file_ids length {len(d['file_ids'])} != X rows {n}"


@pytest.mark.skipif(not HAS_SPECTRO, reason="Spectrogram features not found")
class TestSpectroConsistency:
    def test_sample_count_matches(self):
        d = _load_spectro()
        n = d["X"].shape[0]
        assert len(d["y_drone"]) == n, f"y_drone length {len(d['y_drone'])} != X rows {n}"
        assert len(d["file_ids"]) == n, f"file_ids length {len(d['file_ids'])} != X rows {n}"


@pytest.mark.skipif(not HAS_IQ, reason="IQ features not found")
class TestIqConsistency:
    def test_sample_count_matches(self):
        d = _load_iq()
        n = d["X"].shape[0]
        assert len(d["y_drone"]) == n, f"y_drone length {len(d['y_drone'])} != X rows {n}"
        assert len(d["file_ids"]) == n, f"file_ids length {len(d['file_ids'])} != X rows {n}"


@pytest.mark.skipif(not HAS_ALL_FEATURES, reason="Not all feature types available")
class TestCrossFeatureConsistency:
    def test_same_sample_count(self):
        psd = _load_psd()
        spectro = _load_spectro()
        iq = _load_iq()
        n_psd = psd["X"].shape[0]
        n_spectro = spectro["X"].shape[0]
        n_iq = iq["X"].shape[0]
        assert n_psd == n_spectro == n_iq, (
            f"Sample counts differ: PSD={n_psd}, spectro={n_spectro}, IQ={n_iq}"
        )

    def test_same_file_ids(self):
        psd = _load_psd()
        spectro = _load_spectro()
        iq = _load_iq()
        np.testing.assert_array_equal(
            psd["file_ids"], spectro["file_ids"],
            err_msg="PSD and spectrogram file_ids differ",
        )
        np.testing.assert_array_equal(
            psd["file_ids"], iq["file_ids"],
            err_msg="PSD and IQ file_ids differ",
        )

    def test_same_drone_labels(self):
        psd = _load_psd()
        spectro = _load_spectro()
        iq = _load_iq()
        np.testing.assert_array_equal(
            psd["y_drone"], spectro["y_drone"],
            err_msg="PSD and spectrogram drone labels differ",
        )
        np.testing.assert_array_equal(
            psd["y_drone"], iq["y_drone"],
            err_msg="PSD and IQ drone labels differ",
        )


def _load_any_feature_metadata():
    """Load y_drone and file_ids from whichever feature file is available."""
    if HAS_PSD:
        return _load_psd()
    if HAS_IQ:
        return _load_iq()
    if HAS_SPECTRO:
        return _load_spectro()
    pytest.skip("No feature files available")


HAS_ANY_FEATURES = HAS_PSD or HAS_SPECTRO or HAS_IQ


@pytest.mark.skipif(
    not (HAS_SPLIT and HAS_ANY_FEATURES),
    reason="Split indices or feature files not found",
)
class TestSplitIntegrity:
    @pytest.fixture(autouse=True)
    def _load_data(self):
        self.meta = _load_any_feature_metadata()
        self.split = load_split(SPLIT_PATH)
        self.n_samples = len(self.meta["y_drone"])

    def test_no_data_leakage(self):
        file_ids = self.meta["file_ids"]
        train_files = set(file_ids[self.split["train_idx"]])
        val_files = set(file_ids[self.split["val_idx"]])
        test_files = set(file_ids[self.split["test_idx"]])

        assert not (train_files & val_files), (
            f"File leakage train<->val: {train_files & val_files}"
        )
        assert not (train_files & test_files), (
            f"File leakage train<->test: {train_files & test_files}"
        )
        assert not (val_files & test_files), (
            f"File leakage val<->test: {val_files & test_files}"
        )

    def test_split_proportions(self):
        n_train = len(self.split["train_idx"])
        n_val = len(self.split["val_idx"])
        n_test = len(self.split["test_idx"])
        total = n_train + n_val + n_test

        assert total == self.n_samples, (
            f"Split covers {total} samples but features have {self.n_samples}"
        )

        train_pct = n_train / total
        val_pct = n_val / total
        test_pct = n_test / total

        assert abs(train_pct - 0.70) < 0.05, f"Train proportion {train_pct:.2%} outside 70% +/-5%"
        assert abs(val_pct - 0.15) < 0.05, f"Val proportion {val_pct:.2%} outside 15% +/-5%"
        assert abs(test_pct - 0.15) < 0.05, f"Test proportion {test_pct:.2%} outside 15% +/-5%"

    def test_all_drone_classes_in_each_set(self):
        y = self.meta["y_drone"]
        all_classes = set(np.unique(y))

        for name in ("train_idx", "val_idx", "test_idx"):
            idx = self.split[name]
            partition_classes = set(np.unique(y[idx]))
            missing = all_classes - partition_classes
            assert not missing, (
                f"{name} is missing drone classes: {missing}"
            )

    def test_indices_within_bounds(self):
        for name in ("train_idx", "val_idx", "test_idx"):
            idx = self.split[name]
            assert np.all(idx >= 0), f"{name} contains negative indices"
            assert np.all(idx < self.n_samples), (
                f"{name} contains indices >= {self.n_samples}"
            )

    def test_no_duplicate_indices(self):
        all_idx = np.concatenate([
            self.split["train_idx"],
            self.split["val_idx"],
            self.split["test_idx"],
        ])
        assert len(all_idx) == len(np.unique(all_idx)), "Duplicate indices across partitions"


@pytest.mark.skipif(not HAS_ANY_FEATURES, reason="No feature files available")
class TestSplitCreation:
    def test_create_split_produces_valid_output(self, tmp_path):
        meta = _load_any_feature_metadata()
        y = meta["y_drone"]
        file_ids = meta["file_ids"]
        save_path = tmp_path / "test_split.npz"

        result = create_stratified_split(y, file_ids, save_path=save_path)

        assert "train_idx" in result
        assert "val_idx" in result
        assert "test_idx" in result
        assert save_path.exists()

        loaded = load_split(save_path)
        np.testing.assert_array_equal(result["train_idx"], loaded["train_idx"])
        np.testing.assert_array_equal(result["val_idx"], loaded["val_idx"])
        np.testing.assert_array_equal(result["test_idx"], loaded["test_idx"])
