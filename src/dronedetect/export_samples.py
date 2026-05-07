"""Export test samples for the Streamlit inference interface.

Selects one representative file (100 segments) per (drone, condition) combo
from the test set, and saves per-data-type .npz files organized by condition.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from dronedetect.config import FEATURES_DIR, TEST_SAMPLES_DIR
from dronedetect.splitting import load_split

logger = logging.getLogger(__name__)


def _load_feature_data(features_dir: Path) -> dict:
    """Load all three feature types and their metadata from disk.

    Returns a dict keyed by feature type ("psd", "spectrogram", "iq"),
    each containing X (features array) and metadata arrays.
    Missing feature files are silently skipped.
    """
    result = {}

    psd_path = features_dir / "psd_features.npz"
    if psd_path.exists():
        data = np.load(psd_path)
        result["psd"] = {
            "X": data["X"],
            "y_drone": data["y_drone"],
            "y_interference": data["y_interference"],
            "file_ids": data["file_ids"],
            "drone_classes": data["drone_classes"],
            "interference_classes": data["interference_classes"],
        }
        logger.info("Loaded PSD features: %s", psd_path)

    spec_path = features_dir / "spectrogram_features.npy"
    spec_meta_path = features_dir / "spectrogram_meta.npz"
    if spec_path.exists() and spec_meta_path.exists():
        X_spec = np.load(spec_path, mmap_mode="r")
        meta = np.load(spec_meta_path)
        result["spectrogram"] = {
            "X": X_spec,
            "y_drone": meta["y_drone"],
            "y_interference": meta["y_interference"],
            "file_ids": meta["file_ids"],
            "drone_classes": meta["drone_classes"],
            "interference_classes": meta["interference_classes"],
        }
        logger.info("Loaded spectrogram features: %s", spec_path)

    iq_path = features_dir / "iq_features.npz"
    if iq_path.exists():
        data = np.load(iq_path)
        result["iq"] = {
            "X": data["X"],
            "y_drone": data["y_drone"],
            "y_interference": data["y_interference"],
            "file_ids": data["file_ids"],
            "drone_classes": data["drone_classes"],
            "interference_classes": data["interference_classes"],
        }
        logger.info("Loaded IQ features: %s", iq_path)

    return result


def export_test_samples(
    features_dir: Path,
    split_path: Path,
    output_dir: Path,
) -> None:
    """Export one representative file per (drone, condition) for each feature type.

    Args:
        features_dir: Directory containing pipeline output (.npz/.npy files).
        split_path: Path to the split_indices.npz file.
        output_dir: Root output directory (subdirs per condition will be created).
    """
    features_dir = Path(features_dir)
    split_path = Path(split_path)
    output_dir = Path(output_dir)

    split = load_split(split_path)
    test_idx = split["test_idx"]
    logger.info("Loaded split: %d test samples", len(test_idx))

    all_features = _load_feature_data(features_dir)
    if not all_features:
        logger.warning("No feature files found in %s", features_dir)
        return

    # Use the first available feature type to determine (drone, condition) combos
    ref_key = next(iter(all_features))
    ref = all_features[ref_key]
    drone_classes = ref["drone_classes"]
    interference_classes = ref["interference_classes"]

    y_drone_test = ref["y_drone"][test_idx]
    y_interference_test = ref["y_interference"][test_idx]
    file_ids_test = ref["file_ids"][test_idx]

    combos = set(zip(y_drone_test, y_interference_test))
    logger.info("Found %d (drone, condition) combos in test set", len(combos))

    exported = 0
    for drone_idx, interf_idx in sorted(combos):
        drone_name = str(drone_classes[drone_idx])
        condition_name = str(interference_classes[interf_idx])

        # Find segments matching this combo in the test set
        mask = (y_drone_test == drone_idx) & (y_interference_test == interf_idx)
        combo_file_ids = file_ids_test[mask]

        # Pick the first unique file_id (deterministic)
        unique_files = []
        seen = set()
        for fid in combo_file_ids:
            if fid not in seen:
                unique_files.append(fid)
                seen.add(fid)
        target_file = unique_files[0]

        # Select all segments from that file across the full test set
        file_mask_test = file_ids_test == target_file

        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)

        for feat_type, feat_data in all_features.items():
            X_slice = np.array(feat_data["X"][test_idx[file_mask_test]])
            y_slice = feat_data["y_drone"][test_idx[file_mask_test]]
            y_interf_slice = feat_data["y_interference"][test_idx[file_mask_test]]

            filename = f"{feat_type}_{condition_name}_{drone_name}.npz"
            out_path = condition_dir / filename
            np.savez_compressed(
                out_path,
                X=X_slice,
                y=y_slice,
                y_interference=y_interf_slice,
                drone_class=drone_name,
                interference_class=condition_name,
            )
            logger.info(
                "Exported %s (%d segments, %.1f KB)",
                out_path,
                len(X_slice),
                out_path.stat().st_size / 1024,
            )
            exported += 1

    logger.info("Export complete: %d files written to %s", exported, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export test samples for the Streamlit interface"
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=FEATURES_DIR,
        help="Directory containing pipeline feature files",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="Path to split_indices.npz (default: <features-dir>/../split_indices.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for test samples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    split_path = args.split_path or (args.features_dir.parent / "split_indices.npz")
    output_dir = args.output_dir or TEST_SAMPLES_DIR

    export_test_samples(
        features_dir=args.features_dir,
        split_path=split_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
