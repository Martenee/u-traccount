from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BrackishMOT"
    / "BrackishMOT"
    / "train"
    / "brackishMOT-01"
    / "img1"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "baseline"


def main():
    image_files = sorted(IMAGE_DIRECTORY.glob("*.jpg"))

    if not image_files:
        raise FileNotFoundError(
            f"No JPG images were found in: {IMAGE_DIRECTORY}"
        )

    input_image = image_files[0]
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Small pretrained detector suitable for an initial CPU baseline.
    model = YOLO("yolo11n.pt")

    results = model.predict(
        source=str(input_image),
        conf=0.20,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    output_file = OUTPUT_DIRECTORY / "brackishMOT-01_yolo11.jpg"
    result.save(filename=str(output_file))

    print("Input image:", input_image)
    print("Detections:", len(result.boxes))

    if len(result.boxes) == 0:
        print("No objects detected by the generic pretrained model.")
    else:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = result.names[class_id]
            print(f"- {class_name}: {confidence:.3f}")

    print("Result saved to:", output_file)


if __name__ == "__main__":
    main()