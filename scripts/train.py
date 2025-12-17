# imports
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Union


# Add trainers directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "trainers"))

from trainer import trainer

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------
# Environment Setup
# ---------------------------
def setup_environment() -> dict:
    """
    Set up nnU-Net environment variables.
    Reads from ENV, falls back to default paths.
    """
    paths = {
        "nnUNet_raw": os.getenv("nnUNet_raw", str(Path.cwd() / "Datasets_Tr")),
        "nnUNet_preprocessed": os.getenv("nnUNet_preprocessed", str(Path.cwd() / "Datasets_preprocessed")),
        "nnUNet_results": os.getenv("nnUNet_results", str(Path.cwd() / "Datasets_results")),
    }

    logger.info("Setting up nnU-Net environment variables:")
    for key, value in paths.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)
        logger.info(f"  {key}: {value}")

    return paths


# ---------------------------
# Dataset Checking
# ---------------------------
def check_dataset(dataset_id: int, paths: dict) -> bool:
    """
    Verify that the dataset exists and has the correct structure.

    Args:
        dataset_id: ID of the dataset (e.g., 1 for Dataset001_CT_Scans)
        paths: Dictionary of nnU-Net paths

    Returns:
        True if dataset is ready, False otherwise.
    """
    dataset_name = f"Dataset{dataset_id:03d}_CT_Scans"
    raw_path = Path(paths["nnUNet_raw"]) / dataset_name

    logger.info(f"Checking dataset: {dataset_name}")
    logger.info(f"Location: {raw_path}")

    if not raw_path.exists():
        logger.error(f"Dataset not found at {raw_path}")
        return False

    required_dirs = ["imagesTr", "labelsTr"]
    missing = [d for d in required_dirs if not (raw_path / d).exists()]
    if not (raw_path / "dataset.json").exists():
        missing.append("dataset.json")

    if missing:
        logger.error(f"Missing required files/folders: {', '.join(missing)}")
        return False

    n_images = len(list((raw_path / "imagesTr").glob("*.nii.gz")))
    n_labels = len(list((raw_path / "labelsTr").glob("*.nii.gz")))

    logger.info(f"✓ Dataset found! Images: {n_images}, Labels: {n_labels}")
    if n_images != n_labels:
        logger.warning(f"Number of images ({n_images}) != number of labels ({n_labels})")

    return True


# ---------------------------
# Preprocessing
# ---------------------------
def run_preprocessing(dataset_id: int) -> bool:
    """
    Run nnU-Net preprocessing to create training plans.
    """
    logger.info("="*80)
    logger.info("STEP 1: PREPROCESSING")
    logger.info("="*80)

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-c", "3d_fullres",
        "--verify_dataset_integrity"
    ]

    logger.info(f"Running preprocessing command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=os.environ.copy(), check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        logger.info("✓ Preprocessing completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Preprocessing failed with return code {e.returncode}")
        logger.error(e.stderr)
        return False


# ---------------------------
# Training
# ---------------------------
def run_training(dataset_id: int, fold: Union[int, str] = 0, configuration: str = "3d_fullres") -> bool:
    """
    Run training using the custom trainer.

    Args:
        dataset_id: Dataset ID
        fold: Fold number, 0 for small dataset
        configuration: '2d', '3d_fullres', or '3d_lowres'
    """
    logger.info("="*80)
    logger.info("STEP 2: TRAINING")
    logger.info("="*80)

    dataset_name = f"Dataset{dataset_id:03d}_CT_Scans"
    trainer_name = "trainer"

    cmd = [
        "nnUNetv2_train",
        dataset_name,
        configuration,
        str(fold),
        "-tr", trainer_name,
        "--npz",
        "-device", "cpu"
    ]

    logger.info(f"Running training command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=os.environ.copy(), check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        logger.info("✓ Training completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with return code {e.returncode}")
        logger.error(e.stderr)
        return False


# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="nnInteractive Training Pipeline for Industrial CT Data")
    parser.add_argument("--dataset", type=int, default=1, help="Dataset ID (e.g., 1 for Dataset001_CT_Scans)")
    parser.add_argument("--fold", type=Union[int,str], default=0, help="Fold number or 'all'")
    parser.add_argument("--config", type=str, default="3d_fullres", help="Training configuration (2d/3d_fullres/3d_lowres)")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing if already done")
    args = parser.parse_args()

    paths = setup_environment()

    if not check_dataset(args.dataset, paths):
        logger.error("Dataset check failed. Please fix the issues and try again.")
        return

    # Preprocessing
    if not args.skip_preprocessing:
        preprocessed_dir = Path(paths["nnUNet_preprocessed"]) / f"Dataset{args.dataset:03d}_CT_Scans"
        if preprocessed_dir.exists():
            import shutil
            shutil.rmtree(preprocessed_dir)
            logger.info("Old preprocessed data removed.")
        if not run_preprocessing(args.dataset):
            logger.error("Preprocessing failed. Exiting.")
            return
    else:
        logger.info("Skipping preprocessing (skip_preprocessing=True)")

    # Training
    success = run_training(dataset_id=args.dataset, fold=args.fold, configuration=args.config)

    if success:
        results_path = Path(paths["nnUNet_results"]) / f"Dataset{args.dataset:03d}_CT_Scans" / f"trainer__nnUNetPlans__{args.config}"
        logger.info(f"Results saved to: {results_path}")
        logger.info("To evaluate your model, run:")
        logger.info(f"nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d Dataset{args.dataset:03d}_CT_Scans -tr trainer -c {args.config} -f {args.fold}")
    else:
        logger.error("Training failed. Please check the errors above.")


if __name__ == "__main__":
    main()