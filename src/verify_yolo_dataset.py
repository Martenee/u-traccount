from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brackishmot_yolo"
)

IMAGE_DIRECTORY = DATASET_ROOT / "images" / "train"
LABEL_DIRECTORY = DATASET_ROOT / "labels" / "train"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "yolo_verification"
    / "converted_label_example.jpg"
)


def find_annotated_example():
    label_files = sorted(LABEL_DIRECTORY.glob("*.txt"))

    for label_file in label_files:
        if label_file.read_text(encoding="utf-8").strip():
            image_file = IMAGE_DIRECTORY / f"{label_file.stem}.jpg"

            if image_file.exists():
                return image_file, label_file

    raise FileNotFoundError(
        "No annotated image and label pair was found."
    )


def draw_yolo_labels(image, label_file):
    image_height, image_width = image.shape[:2]
    box_count = 0

    label_lines = label_file.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in label_lines:
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        x_center = float(values[1]) * image_width
        y_center = float(values[2]) * image_height
        box_width = float(values[3]) * image_width
        box_height = float(values[4]) * image_height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            image,
            f"class {class_id}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        box_count += 1

    return box_count


def main():
    image_file, label_file = find_annotated_example()

    image = cv2.imread(str(image_file))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_file}")

    box_count = draw_yolo_labels(image, label_file)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_FILE), image):
        raise RuntimeError("Could not save verification image.")

    print("Image checked:", image_file.name)
    print("Label checked:", label_file.name)
    print("Boxes drawn:", box_count)
    print("Verification image saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()