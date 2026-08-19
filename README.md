# U-TracCount

U-TracCount is an MSc Artificial Intelligence dissertation project investigating how general-purpose object tracking and crowd-counting methods can be adapted to count and track underwater creatures.

## Project objectives

- Detect underwater creatures in images and video.
- Track individual creatures across consecutive video frames.
- Estimate the number of creatures present.
- Compare baseline methods with advanced counting and tracking approaches.
- Evaluate counting using MAE and RMSE.
- Evaluate tracking using metrics such as MOTA and IDF1.

## Planned approach

1. Prepare and inspect underwater datasets.
2. Implement a baseline object detector and tracker.
3. Test open-vocabulary counting using CountGD++.
4. Investigate tracking using COVTrack.
5. Evaluate performance under underwater challenges such as poor visibility, occlusion and colour distortion.

## Project structure

- `configs/` – experiment configuration files
- `data/` – datasets and annotations
- `notebooks/` – exploratory analysis
- `outputs/` – generated results and visualisations
- `src/` – project source code
- `requirements.txt` – Python package requirements

## Environment

- Python 3.10
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib

## Installation

```bash
pip install -r requirements.txt