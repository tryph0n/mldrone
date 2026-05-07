"""CLI for uploading project artifacts to Scaleway S3."""

import argparse
import logging
import sys
from pathlib import Path

from dronedetect.storage import upload_features, upload_models, upload_split

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_DIR = Path("data/features")
DEFAULT_MODELS_DIR = Path("data/models")
DEFAULT_SPLIT_PATH = Path("data/split_indices.npz")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Upload project artifacts to Scaleway S3.",
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help=f"Upload feature files from {DEFAULT_FEATURES_DIR}",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help=f"Upload model files from {DEFAULT_MODELS_DIR}",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=f"Upload split indices from {DEFAULT_SPLIT_PATH}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload features, models, and split (equivalent to --features --models --split)",
    )

    args = parser.parse_args()

    if args.all:
        args.features = True
        args.models = True
        args.split = True

    if not (args.features or args.models or args.split):
        parser.print_usage()
        sys.exit(1)

    if args.features:
        logger.info("Uploading features from %s", DEFAULT_FEATURES_DIR)
        upload_features(DEFAULT_FEATURES_DIR)

    if args.models:
        logger.info("Uploading models from %s", DEFAULT_MODELS_DIR)
        upload_models(DEFAULT_MODELS_DIR)

    if args.split:
        logger.info("Uploading split indices from %s", DEFAULT_SPLIT_PATH)
        upload_split(DEFAULT_SPLIT_PATH)

    logger.info("Upload complete.")


if __name__ == "__main__":
    main()
