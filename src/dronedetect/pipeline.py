"""Streaming feature extraction pipeline for DroneDetect V2.

Downloads .dat files one at a time from S3, extracts PSD, spectrogram, and IQ
features, accumulates results into memmap/arrays, then saves and uploads.

Architecture:
- Background thread prefetches the NEXT file while current one is being processed
- ProcessPoolExecutor handles CPU-bound per-segment feature extraction
- Single load per file: raw IQ loaded once, all 3 feature types extracted from same data
- Spectrogram saved as .npy (memmap-friendly, avoids CRC32 over 11.6 GB)
- PSD and IQ saved as .npz compressed (small arrays)

Peak disk usage: ~12.5 GB (1 .dat file + spectrogram memmap).

Usage:
    uv run python -m dronedetect.pipeline --concurrency 2 --features all
"""

import argparse
import json
import logging
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from queue import Queue
import numpy as np
from tqdm import tqdm

from .config import (
    DEFAULT_IQ_DOWNSAMPLE,
    DEFAULT_NFFT,
    DEFAULT_NOVERLAP,
    DEFAULT_SEGMENT_MS,
    FEATURES_DIR,
    FS,
    IMG_SIZE,
)
from .data_loader import load_raw_iq, parse_filename
from .features import compute_psd, compute_spectrogram
from .preprocessing import downsample_iq, normalize, normalize_minmax
from .storage import (
    download_file,
    get_s3_client,
    list_dat_files,
    upload_features,
)

logger = logging.getLogger(__name__)

SEGMENTS_PER_FILE = 100  # 120M samples / 1.2M per segment = 100
CHECKPOINT_FILE = "pipeline_checkpoint.json"
PSD_MEMMAP_FILE = "_psd_memmap.dat"
IQ_MEMMAP_FILE = "_iq_memmap.dat"
SPEC_MEMMAP_FILE = "_spec_memmap.dat"
LABELS_MEMMAP_FILE = "_labels_memmap.dat"
INTERFERENCE_MEMMAP_FILE = "_interference_memmap.dat"
STATE_MEMMAP_FILE = "_state_memmap.dat"
FILEIDS_MEMMAP_FILE = "_fileids_memmap.dat"


def segment_signal_lazy(signal: np.ndarray, segment_ms: float, fs: float):
    """Yield segment views without pre-allocating the full array.

    Each yielded segment is a numpy view (no copy) into the original signal,
    saving ~2 GB RAM compared to np.array_split which allocates all at once.
    """
    samples_per_segment = int(segment_ms / 1e3 * fs)
    n_segments = len(signal) // samples_per_segment
    for i in range(n_segments):
        start = i * samples_per_segment
        yield signal[start : start + samples_per_segment]


def _compute_total_segments(n_files: int) -> int:
    """Total expected segments across all files."""
    return n_files * SEGMENTS_PER_FILE


def _prefetch_worker(
    keys: list[str],
    start_idx: int,
    tmp_dir: Path,
    queue: Queue,
    client,
) -> None:
    """Background thread: downloads files and puts paths into queue."""
    for i in range(start_idx, len(keys)):
        key = keys[i]
        filename = Path(key).name
        local_path = tmp_dir / filename
        try:
            download_file(key, local_path, client=client)
            queue.put((i, key, local_path, None))
        except Exception as exc:
            queue.put((i, key, None, exc))
    queue.put(None)  # Sentinel: no more files


def _build_file_manifest(keys: list[str]) -> list[dict]:
    """Build deterministic file manifest with metadata from S3 keys.

    Each entry contains the file_id (global segment offset), key, and parsed metadata.
    Files are already sorted by key (done in list_dat_files).
    """
    manifest = []
    segment_offset = 0
    for file_id, key in enumerate(keys):
        filename = Path(key).name
        meta = parse_filename(filename)
        manifest.append(
            {
                "key": key,
                "filename": filename,
                "file_id": file_id,
                "segment_offset": segment_offset,
                "drone_code": meta["drone_code"],
                "interference": meta["interference"],
                "state": meta["state"],
            }
        )
        segment_offset += SEGMENTS_PER_FILE
    return manifest


def _load_checkpoint(features_dir: Path) -> set[str]:
    """Load set of already-processed file keys from checkpoint."""
    cp_path = features_dir / CHECKPOINT_FILE
    if cp_path.exists():
        data = json.loads(cp_path.read_text())
        return set(data.get("processed_keys", []))
    return set()


def _save_checkpoint(features_dir: Path, processed_keys: set[str]) -> None:
    """Persist processed keys for resume capability."""
    cp_path = features_dir / CHECKPOINT_FILE
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    cp_path.write_text(json.dumps({"processed_keys": sorted(processed_keys)}))


def _process_segment(
    seg_data: np.ndarray,
    features_set: set[str],
) -> dict:
    """Process a single segment: normalize and extract requested features.

    Runs in a worker process via ProcessPoolExecutor. Returns a dict of computed
    feature arrays, keyed by feature name. The caller writes results into memmaps.
    """
    result = {}
    need_zscore = "psd" in features_set or "spectrogram" in features_set

    if need_zscore:
        seg_norm = normalize(seg_data)

    if "psd" in features_set:
        _, psd = compute_psd(seg_norm, fs=FS, nfft=DEFAULT_NFFT)
        psd_norm = psd / (psd.max() + 1e-12)
        result["psd"] = psd_norm.astype(np.float32)

    if "spectrogram" in features_set:
        spec_rgb = compute_spectrogram(
            seg_norm,
            fs=FS,
            nfft=DEFAULT_NFFT,
            noverlap=DEFAULT_NOVERLAP,
            target_size=IMG_SIZE,
        )
        result["spectrogram"] = spec_rgb

    if "iq" in features_set:
        seg_minmax = normalize_minmax(seg_data)
        iq_down = downsample_iq(seg_minmax, target_samples=DEFAULT_IQ_DOWNSAMPLE)
        result["iq"] = iq_down.astype(np.float32)

    return result


def _extract_features_for_file(
    local_path: Path,
    segment_offset: int,
    psd_array: np.ndarray,
    spec_memmap: np.memmap,
    iq_array: np.ndarray,
    labels: np.ndarray,
    label_id: int,
    interference_labels: np.ndarray,
    interference_id: int,
    state_labels: np.ndarray,
    state_id: int,
    file_ids: np.ndarray,
    file_id: int,
    features_set: set[str],
    concurrency: int = 2,
) -> None:
    """Process one .dat file: load once, segment lazily, extract features.

    Uses ProcessPoolExecutor for CPU-bound per-segment feature extraction.
    Writes directly into pre-allocated memmap arrays at the correct offset.
    """
    raw_iq = load_raw_iq(local_path)

    # Collect segments into a list for parallel dispatch.
    # Each segment is a numpy view (no copy) from segment_signal_lazy.
    # We must materialize them before sending to worker processes since views
    # cannot survive pickling — np.array(seg) creates a contiguous copy only
    # at dispatch time, keeping peak RAM to (raw + N_workers * segment_size).
    segments = list(segment_signal_lazy(raw_iq, segment_ms=DEFAULT_SEGMENT_MS, fs=FS))
    n_seg = min(len(segments), SEGMENTS_PER_FILE)

    def _write_metadata(global_idx: int) -> None:
        labels[global_idx] = label_id
        interference_labels[global_idx] = interference_id
        state_labels[global_idx] = state_id
        file_ids[global_idx] = file_id

    if concurrency > 1:
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(n_seg):
                # np.ascontiguousarray ensures picklable data for worker processes
                seg_copy = np.ascontiguousarray(segments[i])
                future = executor.submit(_process_segment, seg_copy, features_set)
                futures.append((i, future))

            for i, future in futures:
                global_idx = segment_offset + i
                result = future.result()

                if "psd" in result and psd_array is not None:
                    psd_array[global_idx] = result["psd"]
                if "spectrogram" in result and spec_memmap is not None:
                    spec_memmap[global_idx] = result["spectrogram"]
                if "iq" in result and iq_array is not None:
                    iq_array[global_idx] = result["iq"]
                _write_metadata(global_idx)
    else:
        # Sequential fallback (concurrency=1): no process overhead
        for i in range(n_seg):
            global_idx = segment_offset + i
            result = _process_segment(segments[i], features_set)

            if "psd" in result and psd_array is not None:
                psd_array[global_idx] = result["psd"]
            if "spectrogram" in result and spec_memmap is not None:
                spec_memmap[global_idx] = result["spectrogram"]
            if "iq" in result and iq_array is not None:
                iq_array[global_idx] = result["iq"]
            _write_metadata(global_idx)

    del raw_iq, segments


def _build_label_encoder(manifest: list[dict]) -> dict[str, int]:
    """Build deterministic drone_code -> integer label mapping."""
    codes = sorted(set(entry["drone_code"] for entry in manifest))
    return {code: idx for idx, code in enumerate(codes)}


def _build_interference_encoder(manifest: list[dict]) -> dict[str, int]:
    """Build deterministic interference condition -> integer mapping."""
    conditions = sorted(set(entry["interference"] for entry in manifest))
    return {cond: idx for idx, cond in enumerate(conditions)}


def _build_state_encoder(manifest: list[dict]) -> dict[str, int]:
    """Build deterministic drone state -> integer mapping."""
    states = sorted(set(entry["state"] for entry in manifest))
    return {state: idx for idx, state in enumerate(states)}


def run_pipeline(
    features_to_extract: set[str],
    concurrency: int = 2,
    output_dir: Path | None = None,
    upload: bool = False,
    tmp_dir: Path | None = None,
    max_files: int | None = None,
) -> None:
    """Execute the full feature extraction pipeline.

    Args:
        features_to_extract: Subset of {"psd", "spectrogram", "iq"}
        concurrency: Number of ProcessPoolExecutor workers for feature extraction
        output_dir: Where to write feature files (default: FEATURES_DIR)
        upload: Whether to upload results to S3 after completion
        tmp_dir: Temporary directory for .dat downloads (default: system temp)
        max_files: If set, process only the first N files (for testing)
    """
    if output_dir is None:
        output_dir = FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to S3 and listing files...")
    client = get_s3_client()
    keys = list_dat_files(client=client)

    if not keys:
        logger.error("No .dat files found on S3. Aborting.")
        return

    manifest = _build_file_manifest(keys)

    if max_files is not None:
        logger.warning(
            "--max-files=%d: truncating manifest from %d to %d files (testing mode)",
            max_files,
            len(manifest),
            min(max_files, len(manifest)),
        )
        manifest = manifest[:max_files]

    total_segments = _compute_total_segments(len(manifest))
    label_encoder = _build_label_encoder(manifest)
    interference_encoder = _build_interference_encoder(manifest)
    state_encoder = _build_state_encoder(manifest)

    logger.info(
        "Pipeline: %d files, %d segments, features=%s, concurrency=%d",
        len(manifest),
        total_segments,
        features_to_extract,
        concurrency,
    )
    logger.info("Drone encoding: %s", label_encoder)
    logger.info("Interference encoding: %s", interference_encoder)
    logger.info("State encoding: %s", state_encoder)

    # Resume support
    processed_keys = _load_checkpoint(output_dir)
    if processed_keys:
        logger.info("Resuming: %d files already processed", len(processed_keys))

    # Allocate accumulators as memmaps (persist across interruptions)
    psd_array: np.memmap | None = None
    spec_memmap: np.memmap | None = None
    iq_array: np.memmap | None = None

    memmap_mode = "r+" if processed_keys else "w+"

    labels_path = output_dir / LABELS_MEMMAP_FILE
    labels = np.memmap(
        labels_path, dtype=np.int64, mode=memmap_mode, shape=(total_segments,)
    )
    interference_path = output_dir / INTERFERENCE_MEMMAP_FILE
    interference_labels = np.memmap(
        interference_path, dtype=np.int64, mode=memmap_mode, shape=(total_segments,)
    )
    state_path = output_dir / STATE_MEMMAP_FILE
    state_labels = np.memmap(
        state_path, dtype=np.int64, mode=memmap_mode, shape=(total_segments,)
    )
    fileids_path = output_dir / FILEIDS_MEMMAP_FILE
    file_ids = np.memmap(
        fileids_path, dtype=np.int64, mode=memmap_mode, shape=(total_segments,)
    )

    if "psd" in features_to_extract:
        psd_path_mm = output_dir / PSD_MEMMAP_FILE
        psd_shape = (total_segments, DEFAULT_NFFT)
        psd_array = np.memmap(
            psd_path_mm, dtype=np.float32, mode=memmap_mode, shape=psd_shape
        )
        logger.info("PSD memmap: %.1f MB", psd_array.nbytes / 1e6)

    if "spectrogram" in features_to_extract:
        spec_mm_path = output_dir / SPEC_MEMMAP_FILE
        spec_shape = (total_segments, IMG_SIZE[0], IMG_SIZE[1], 3)
        spec_memmap = np.memmap(
            spec_mm_path, dtype=np.float32, mode=memmap_mode, shape=spec_shape
        )
        logger.info("Spectrogram memmap: %.1f GB", spec_memmap.nbytes / 1e9)

    if "iq" in features_to_extract:
        iq_path_mm = output_dir / IQ_MEMMAP_FILE
        iq_shape = (total_segments, 2, DEFAULT_IQ_DOWNSAMPLE)
        iq_array = np.memmap(
            iq_path_mm, dtype=np.float32, mode=memmap_mode, shape=iq_shape
        )
        logger.info("IQ memmap: %.1f GB", iq_array.nbytes / 1e9)

    # Determine which files still need processing
    files_to_process = [
        entry for entry in manifest if entry["key"] not in processed_keys
    ]
    keys_to_download = [entry["key"] for entry in files_to_process]

    if not files_to_process:
        logger.info("All files already processed. Skipping to save step.")
    else:
        # Setup prefetch (I/O-bound: background thread downloads next file)
        use_tmp = tmp_dir or Path(tempfile.mkdtemp(prefix="dronedetect_"))
        use_tmp.mkdir(parents=True, exist_ok=True)

        prefetch_queue: Queue = Queue(maxsize=2)
        key_to_entry = {entry["key"]: entry for entry in files_to_process}

        prefetch_thread = threading.Thread(
            target=_prefetch_worker,
            args=(keys_to_download, 0, use_tmp, prefetch_queue, client),
            daemon=True,
        )
        prefetch_thread.start()

        progress = tqdm(
            total=len(files_to_process),
            desc="Processing files",
            unit="file",
        )

        while True:
            item = prefetch_queue.get()
            if item is None:
                break

            dl_idx, key, local_path, error = item

            if error is not None:
                logger.error("Download failed for %s: %s", key, error)
                progress.update(1)
                continue

            entry = key_to_entry[key]
            label_id = label_encoder[entry["drone_code"]]
            interference_id = interference_encoder[entry["interference"]]
            state_id = state_encoder[entry["state"]]

            try:
                _extract_features_for_file(
                    local_path=local_path,
                    segment_offset=entry["segment_offset"],
                    psd_array=psd_array,
                    spec_memmap=spec_memmap,
                    iq_array=iq_array,
                    labels=labels,
                    label_id=label_id,
                    interference_labels=interference_labels,
                    interference_id=interference_id,
                    state_labels=state_labels,
                    state_id=state_id,
                    file_ids=file_ids,
                    file_id=entry["file_id"],
                    features_set=features_to_extract,
                    concurrency=concurrency,
                )
                processed_keys.add(key)
                _save_checkpoint(output_dir, processed_keys)
            except Exception:
                logger.exception("Processing failed for %s", key)
            finally:
                if local_path and local_path.exists():
                    local_path.unlink()
                progress.update(1)

        progress.close()
        prefetch_thread.join(timeout=10)

    # Flush all memmaps before saving
    if psd_array is not None:
        psd_array.flush()
    if spec_memmap is not None:
        spec_memmap.flush()
    if iq_array is not None:
        iq_array.flush()
    labels.flush()
    interference_labels.flush()
    state_labels.flush()
    file_ids.flush()

    # Build class name arrays (sorted, matching encoder order)
    drone_classes = np.array(sorted(label_encoder.keys()), dtype="U32")
    interference_classes = np.array(sorted(interference_encoder.keys()), dtype="U32")
    state_classes = np.array(sorted(state_encoder.keys()), dtype="U32")

    # Materialize metadata memmaps as regular arrays for saving
    y_drone = np.array(labels)
    y_interference = np.array(interference_labels)
    y_state = np.array(state_labels)
    file_ids_np = np.array(file_ids)

    # Shared metadata dict for all feature files
    shared_meta = dict(
        y_drone=y_drone,
        y_interference=y_interference,
        y_state=y_state,
        file_ids=file_ids_np,
        drone_classes=drone_classes,
        interference_classes=interference_classes,
        state_classes=state_classes,
    )

    if "psd" in features_to_extract and psd_array is not None:
        psd_path = output_dir / "psd_features.npz"
        np.savez_compressed(psd_path, X=np.array(psd_array), **shared_meta)
        logger.info("Saved %s (%.1f MB)", psd_path, psd_path.stat().st_size / 1e6)

    if "spectrogram" in features_to_extract and spec_memmap is not None:
        # np.save streams from memmap without loading into RAM (no CRC32 overhead).
        # Metadata saved separately as a companion .npz file.
        spec_path = output_dir / "spectrogram_features.npy"
        np.save(spec_path, spec_memmap)
        logger.info("Saved %s (%.1f GB)", spec_path, spec_path.stat().st_size / 1e9)

        spec_meta_path = output_dir / "spectrogram_meta.npz"
        np.savez_compressed(spec_meta_path, **shared_meta)
        logger.info("Saved spectrogram metadata: %s", spec_meta_path)

    if "iq" in features_to_extract and iq_array is not None:
        iq_path = output_dir / "iq_features.npz"
        np.savez_compressed(iq_path, X=np.array(iq_array), **shared_meta)
        logger.info("Saved %s (%.1f GB)", iq_path, iq_path.stat().st_size / 1e9)

    # Clean up memmaps
    del psd_array, spec_memmap, iq_array, labels
    del interference_labels, state_labels, file_ids
    for mm_file in [
        PSD_MEMMAP_FILE,
        SPEC_MEMMAP_FILE,
        IQ_MEMMAP_FILE,
        LABELS_MEMMAP_FILE,
        INTERFERENCE_MEMMAP_FILE,
        STATE_MEMMAP_FILE,
        FILEIDS_MEMMAP_FILE,
    ]:
        mm_path = output_dir / mm_file
        if mm_path.exists():
            mm_path.unlink()

    # Save manifest for reproducibility
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": manifest,
                "drone_encoder": label_encoder,
                "interference_encoder": interference_encoder,
                "state_encoder": state_encoder,
                "total_segments": total_segments,
                "segments_per_file": SEGMENTS_PER_FILE,
                "config": {
                    "fs": FS,
                    "nfft": DEFAULT_NFFT,
                    "noverlap": DEFAULT_NOVERLAP,
                    "segment_ms": DEFAULT_SEGMENT_MS,
                    "iq_downsample": DEFAULT_IQ_DOWNSAMPLE,
                    "img_size": list(IMG_SIZE),
                },
            },
            indent=2,
        )
    )
    logger.info("Saved manifest: %s", manifest_path)

    # Remove checkpoint after successful completion
    cp_path = output_dir / CHECKPOINT_FILE
    if cp_path.exists():
        cp_path.unlink()
        logger.info("Pipeline complete. Checkpoint removed.")

    if upload:
        logger.info("Uploading features to S3...")
        upload_features(output_dir, client=client)
        logger.info("Upload complete.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DroneDetect V2 streaming feature extraction pipeline",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        help="Features to extract: all, psd, spectrogram, iq (comma-separated)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of parallel workers for feature extraction (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for feature files (default: DRONEDETECT_FEATURES_DIR)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to S3 after extraction",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default=None,
        help="Temporary directory for downloads (default: system temp)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only the first N files (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.features == "all":
        features_set = {"psd", "spectrogram", "iq"}
    else:
        features_set = {f.strip() for f in args.features.split(",")}

    valid_features = {"psd", "spectrogram", "iq"}
    invalid = features_set - valid_features
    if invalid:
        logger.error("Invalid features: %s. Valid: %s", invalid, valid_features)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else None

    run_pipeline(
        features_to_extract=features_set,
        concurrency=args.concurrency,
        output_dir=output_dir,
        upload=args.upload,
        tmp_dir=tmp_dir,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
