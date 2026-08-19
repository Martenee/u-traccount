import configparser
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BrackishMOT"
    / "BrackishMOT"
    / "train"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brackishmot_yolo"
)

CONFIG_FILE = PROJECT_ROOT / "configs" / "brackishmot.yaml"

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
CLASS_ID = 0
CLASS_NAME = "underwater_creature"


def read_sequence_information(sequence_directory):
    config = configparser.ConfigParser()
    config.read(sequence_directory / "seqinfo.ini")

    sequence = config["Sequence"]

    return {
        "width": int(sequence["imWidth"]),
        "height": int(sequence["imHeight"]),
        "extension": sequence.get("imExt", ".jpg"),
    }


def read_annotations(annotation_file):
    annotations_by_frame = defaultdict(list)

    with annotation_file.open("r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 6:
                continue

            frame_number = int(float(row[0]))
            x = float(row[2])
            y = float(row[3])
            width = float(row[4])
            height = float(row[5])

            # Column 7 is the MOT confidence/validity marker.
            if len(row) >= 7 and float(row[6]) <= 0:
                continue

            if width <= 0 or height <= 0:
                continue

            annotations_by_frame[frame_number].append(
                (x, y, width, height)
            )

    return annotations_by_frame


def convert_box_to_yolo(box, image_width, image_height):
    x, y, width, height = box

    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalised_width = width / image_width
    normalised_height = height / image_height

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    normalised_width = min(max(normalised_width, 0.0), 1.0)
    normalised_height = min(max(normalised_height, 0.0), 1.0)

    return (
        x_center,
        y_center,
        normalised_width,
        normalised_height,
    )


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert_sequence(sequence_directory, split):
    information = read_sequence_information(sequence_directory)

    annotations = read_annotations(
        sequence_directory / "gt" / "gt.txt"
    )

    image_directory = sequence_directory / "img1"
    image_files = sorted(
        image_directory.glob(f"*{information['extension']}")
    )

    output_image_directory = OUTPUT_ROOT / "images" / split
    output_label_directory = OUTPUT_ROOT / "labels" / split

    output_image_directory.mkdir(parents=True, exist_ok=True)
    output_label_directory.mkdir(parents=True, exist_ok=True)

    box_count = 0

    for image_file in image_files:
        frame_number = int(image_file.stem)
        output_stem = f"{sequence_directory.name}_{image_file.stem}"

        output_image = (
            output_image_directory
            / f"{output_stem}{image_file.suffix}"
        )
        output_label = output_label_directory / f"{output_stem}.txt"

        link_or_copy(image_file, output_image)

        label_lines = []

        for box in annotations.get(frame_number, []):
            yolo_box = convert_box_to_yolo(
                box,
                information["width"],
                information["height"],
            )

            label_lines.append(
                f"{CLASS_ID} "
                f"{yolo_box[0]:.6f} "
                f"{yolo_box[1]:.6f} "
                f"{yolo_box[2]:.6f} "
                f"{yolo_box[3]:.6f}"
            )
            box_count += 1

        output_label.write_text(
            "\n".join(label_lines),
            encoding="utf-8",
        )

    return len(image_files), box_count


def write_dataset_configuration():
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    configuration = (
        f"path: {OUTPUT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )

    CONFIG_FILE.write_text(configuration, encoding="utf-8")


def main():
    if not SOURCE_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {SOURCE_DIRECTORY}"
        )

    sequences = sorted(
        directory
        for directory in SOURCE_DIRECTORY.iterdir()
        if directory.is_dir()
    )

    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(sequences)

    validation_count = max(
        1,
        round(len(sequences) * VALIDATION_FRACTION),
    )

    validation_sequences = set(sequences[:validation_count])

    totals = {
        "train_sequences": 0,
        "val_sequences": 0,
        "train_images": 0,
        "val_images": 0,
        "train_boxes": 0,
        "val_boxes": 0,
    }

    for sequence_directory in sequences:
        split = (
            "val"
            if sequence_directory in validation_sequences
            else "train"
        )

        image_count, box_count = convert_sequence(
            sequence_directory,
            split,
        )

        totals[f"{split}_sequences"] += 1
        totals[f"{split}_images"] += image_count
        totals[f"{split}_boxes"] += box_count

        print(
            f"Converted {sequence_directory.name}: "
            f"{image_count} images, {box_count} boxes -> {split}"
        )

    write_dataset_configuration()

    print("\nConversion complete")
    print("Training sequences:", totals["train_sequences"])
    print("Validation sequences:", totals["val_sequences"])
    print("Training images:", totals["train_images"])
    print("Validation images:", totals["val_images"])
    print("Training boxes:", totals["train_boxes"])
    print("Validation boxes:", totals["val_boxes"])
    print("Dataset configuration:", CONFIG_FILE)


if __name__ == "__main__":
    main()