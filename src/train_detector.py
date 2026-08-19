import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = PROJECT_ROOT / "configs" / "brackishmot.yaml"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "training"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 on the converted BrackishMOT dataset."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device: cpu, 0, 1, etc.",
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training dataset to use.",
    )

    parser.add_argument(
        "--name",
        default="brackishmot_yolo11n",
        help="Name of the training run.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_CONFIG}"
        )

    if not 0 < arguments.fraction <= 1:
        raise ValueError("--fraction must be greater than 0 and at most 1.")

    print("Dataset configuration:", DATA_CONFIG)
    print("Epochs:", arguments.epochs)
    print("Batch size:", arguments.batch)
    print("Device:", arguments.device)
    print("Dataset fraction:", arguments.fraction)

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_CONFIG),
        epochs=arguments.epochs,
        imgsz=640,
        batch=arguments.batch,
        device=arguments.device,
        workers=0,
        fraction=arguments.fraction,
        project=str(OUTPUT_DIRECTORY),
        name=arguments.name,
        exist_ok=True,
        pretrained=True,
        patience=10,
        seed=42,
        deterministic=True,
        cache=False,
        plots=True,
        amp=arguments.device != "cpu",
    )

    print("Training completed.")
    print(
        "Results saved to:",
        OUTPUT_DIRECTORY / arguments.name,
    )


if __name__ == "__main__":
    main()