from configparser import ConfigParser
from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BrackishMOT"
    / "BrackishMOT"
    / "train"
    / "brackishMOT-01"
)

IMAGE_DIR = SEQUENCE_DIR / "img1"
GT_FILE = SEQUENCE_DIR / "gt" / "gt.txt"
SEQINFO_FILE = SEQUENCE_DIR / "seqinfo.ini"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "brackishMOT-01_frame1.jpg"


def main():
    # Read sequence information
    config = ConfigParser()
    config.read(SEQINFO_FILE)
    sequence_info = config["Sequence"]

    print("Sequence name:", sequence_info.get("name"))
    print("Frame rate:", sequence_info.get("frameRate"))
    print("Sequence length:", sequence_info.get("seqLength"))
    print(
        "Image size:",
        f"{sequence_info.get('imWidth')} x {sequence_info.get('imHeight')}",
    )

    # Find image frames
    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    print("Image frames found:", len(image_files))

    # Read MOT ground-truth annotations
    ground_truth = pd.read_csv(GT_FILE, header=None)
    print("Ground-truth rows:", len(ground_truth))
    print("Annotation columns:", len(ground_truth.columns))
    print("Unique tracked objects:", ground_truth[1].nunique())

    # Load the first frame
    first_frame_path = image_files[0]
    frame = cv2.imread(str(first_frame_path))

    if frame is None:
        raise FileNotFoundError(f"Could not read {first_frame_path}")

    frame_number = int(first_frame_path.stem)
    frame_annotations = ground_truth[ground_truth[0] == frame_number]

    # Draw ground-truth bounding boxes
    for _, row in frame_annotations.iterrows():
        object_id = int(row[1])
        x, y, width, height = map(int, row[2:6])

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"ID {object_id}",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_FILE), frame)

    print("Objects in first frame:", len(frame_annotations))
    print("Annotated frame saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()