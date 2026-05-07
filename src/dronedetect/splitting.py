"""File-level stratified splitting for RF drone classification.

Ensures zero data leakage by grouping segments from the same recording file
into the same partition. Stratification is by drone label only.
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


def create_stratified_split(
    y: np.ndarray,
    file_ids: np.ndarray,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> dict:
    """
    Two-stage stratified group split into train/val/test partitions.

    Stage 1: split off test set using StratifiedGroupKFold.
    Stage 2: split remaining trainval into train and val.

    Stratification is by drone label (y). Grouping is by file_ids to prevent
    segments from the same recording appearing in different partitions.

    Args:
        y: Drone labels array, shape (n_samples,).
        file_ids: File identifier per sample, shape (n_samples,).
        train_size: Fraction of data for training.
        val_size: Fraction of data for validation.
        test_size: Fraction of data for testing.
        random_state: Seed for reproducibility.
        save_path: If provided, save split indices to this .npz path.

    Returns:
        Dict with keys: train_idx, val_idx, test_idx, split_metadata.

    Raises:
        ValueError: If size fractions do not sum to 1 or leakage is detected.
    """
    y = np.asarray(y)
    file_ids = np.asarray(file_ids)

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.4f}")

    # Stage 1: split off test set
    n_splits_test = round(1 / test_size)
    sgkf_test = StratifiedGroupKFold(
        n_splits=n_splits_test, shuffle=True, random_state=random_state
    )
    all_indices = np.arange(len(y))

    trainval_idx, test_idx = next(sgkf_test.split(all_indices, y, groups=file_ids))

    # Stage 2: split trainval into train and val
    relative_val = val_size / (train_size + val_size)
    n_splits_val = round(1 / relative_val)
    sgkf_val = StratifiedGroupKFold(
        n_splits=n_splits_val, shuffle=True, random_state=random_state + 1
    )

    y_trainval = y[trainval_idx]
    file_ids_trainval = file_ids[trainval_idx]

    train_local, val_local = next(
        sgkf_val.split(
            np.arange(len(trainval_idx)), y_trainval, groups=file_ids_trainval
        )
    )
    train_idx = trainval_idx[train_local]
    val_idx = trainval_idx[val_local]

    # Anti-leakage assertion: zero file overlap between all 3 partitions
    train_files = set(file_ids[train_idx])
    val_files = set(file_ids[val_idx])
    test_files = set(file_ids[test_idx])

    overlap_tv = train_files & val_files
    overlap_tt = train_files & test_files
    overlap_vt = val_files & test_files

    if overlap_tv or overlap_tt or overlap_vt:
        msg = "Data leakage detected! File overlap between partitions:"
        if overlap_tv:
            msg += f"\n  train-val: {overlap_tv}"
        if overlap_tt:
            msg += f"\n  train-test: {overlap_tt}"
        if overlap_vt:
            msg += f"\n  val-test: {overlap_vt}"
        raise ValueError(msg)

    logger.info("Split created with zero file overlap (leakage-free)")
    logger.info(
        "Samples  -> train=%d, val=%d, test=%d (total=%d)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        len(y),
    )
    logger.info(
        "Files    -> train=%d, val=%d, test=%d",
        len(train_files),
        len(val_files),
        len(test_files),
    )

    split_metadata = {
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
        "random_state": random_state,
    }

    result = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_metadata": split_metadata,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )
        logger.info("Split indices saved to %s", save_path)

    return result


def verify_split_balance(
    file_ids: np.ndarray,
    y: np.ndarray,
    split_indices: dict,
    conditions: np.ndarray | None = None,
) -> None:
    """
    Log class and condition distributions per partition for diagnostics.

    Args:
        file_ids: File identifier per sample, shape (n_samples,).
        y: Drone labels array, shape (n_samples,).
        split_indices: Dict with train_idx, val_idx, test_idx.
        conditions: Optional condition labels (e.g. CLEAN/WIFI/BLUE/BOTH).
    """
    y = np.asarray(y)
    file_ids = np.asarray(file_ids)
    if conditions is not None:
        conditions = np.asarray(conditions)

    partitions = {
        "train": split_indices["train_idx"],
        "val": split_indices["val_idx"],
        "test": split_indices["test_idx"],
    }

    for name, idx in partitions.items():
        y_part = y[idx]
        classes, counts = np.unique(y_part, return_counts=True)
        dist_str = ", ".join(f"{c}={n}" for c, n in zip(classes, counts))
        logger.info("[%s] Class distribution: %s", name, dist_str)

    if conditions is not None:
        all_classes = np.unique(y)
        all_conditions = np.unique(conditions)

        for name, idx in partitions.items():
            cond_part = conditions[idx]
            conds, counts = np.unique(cond_part, return_counts=True)
            dist_str = ", ".join(f"{c}={n}" for c, n in zip(conds, counts))
            logger.info("[%s] Condition distribution: %s", name, dist_str)

            # Warn if any drone is missing a condition entirely
            y_part = y[idx]
            for drone in all_classes:
                drone_mask = y_part == drone
                drone_conditions = set(np.unique(cond_part[drone_mask]))
                missing = set(all_conditions) - drone_conditions
                if missing:
                    logger.warning(
                        "[%s] Drone '%s' missing conditions: %s",
                        name,
                        drone,
                        missing,
                    )


def load_split(path: str | Path) -> dict:
    """
    Load previously saved split indices from a .npz file.

    Args:
        path: Path to the .npz file created by create_stratified_split.

    Returns:
        Dict with keys: train_idx, val_idx, test_idx.
    """
    data = np.load(path)
    return {
        "train_idx": data["train_idx"],
        "val_idx": data["val_idx"],
        "test_idx": data["test_idx"],
    }
