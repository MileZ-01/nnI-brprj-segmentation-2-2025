# imports
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Union

# Add trainers directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "trainers"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def setup_environment() -> dict:
    """
    Set up nnU-Net environment variables.
    """
    paths = {
        "nnUNet_raw": os.getenv("nnUNet_raw", str(Path.cwd() / "Datasets_Tr")),
        "nnUNet_preprocessed": os.getenv("nnUNet_preprocessed", str(Path.cwd() / "Datasets_preprocessed")),
        "nnUNet_results": os.getenv("nnUNet_results", str(Path.cwd() / "Datasets_results")),
    }
    for key, value in paths.items():
        os.environ[key] = value
        Path(value).mkdir(parents=True, exist_ok=True)
    return paths


def run_evaluation(dataset_id: int,
                   fold: Union[int, str] = 0,
                   configuration: str = "3d_fullres",
                   trainer_name: str = "trainer",
                   input_folder: str = None,
                   output_folder: str = None) -> bool:
    """
    Run evaluation / prediction using a trained nnInteractive model.

    Args:
        dataset_id: Dataset ID (e.g., 1 for Dataset001_CT_Scans)
        fold: Fold number
        configuration: '2d', '3d_fullres', or '3d_lowres'
        trainer_name: Name of the trainer class
        input_folder: Folder with images to predict
        output_folder: Folder to save predictions
    """
    import subprocess

    dataset_name = f"Dataset{dataset_id:03d}_CT_Scans"

    if input_folder is None:
        input_folder = str(Path(os.environ["nnUNet_raw"]) / dataset_name / "imagesTr")
    if output_folder is None:
        output_folder = str(Path(os.environ["nnUNet_results"]) / dataset_name / "predictions")

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-d", dataset_name,
        "-tr", trainer_name,
        "-c", configuration,
        "-f", str(fold)
    ]

    logger.info(f"Running evaluation command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=os.environ.copy(), check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        logger.info(f"✓ Evaluation complete! Predictions saved to {output_folder}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with return code {e.returncode}")
        logger.error(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnInteractive model on CT data")
    parser.add_argument("--dataset", type=int, default=1, help="Dataset ID")
    parser.add_argument("--fold", type=Union[int,str], default=0, help="Fold number")
    parser.add_argument("--config", type=str, default="3d_fullres", help="Training configuration")
    parser.add_argument("--trainer", type=str, default="trainer", help="Trainer class name")
    parser.add_argument("--input", type=str, default=None, help="Input folder for images")
    parser.add_argument("--output", type=str, default=None, help="Output folder for predictions")
    args = parser.parse_args()

    setup_environment()

    success = run_evaluation(
        dataset_id=args.dataset,
        fold=args.fold,
        configuration=args.config,
        trainer_name=args.trainer,
        input_folder=args.input,
        output_folder=args.output
    )

    if not success:
        logger.error("Evaluation failed.")


if __name__ == "__main__":
    main()