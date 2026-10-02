# U-TracCount

U-TracCount is an MSc Artificial Intelligence dissertation project investigating how general-purpose object tracking and crowd-counting methods can be adapted to count and track underwater creatures.

## Project objectives

- Detect underwater creatures in images and video.
- Track individual creatures across consecutive video frames.
- Estimate the number of creatures present.
- Compare baseline methods with advanced counting and tracking approaches.
- Evaluate counting using MAE and RMSE.
- Evaluate tracking using metrics such as MOTA and IDF1.

## Implemented workflow

1. Download and inspect the BrackishMOT dataset.
2. Convert MOTChallenge annotations into YOLO format.
3. Fine-tune and evaluate a YOLO11n underwater-creature detector.
4. Apply ByteTrack to associate detections across consecutive frames.
5. Evaluate tracking using HOTA, CLEAR and identity metrics through TrackEval.
6. Estimate unique-individual counts from sequence-level track identities.
7. Assess CountGD++ as an exploratory prompt-conditioned counting extension.
8. Validate a ULGF-based compositing pipeline as a controlled exploratory diagnostic.

## Project structure

- `configs/` – dataset and ByteTrack configuration files.
- `src/` – BrackishMOT conversion, verification, detector inference and YOLO training scripts.
- `notebooks/` – documented Colab notebook for the complete YOLO–ByteTrack quantitative pipeline.
- `CountGD_experiment/` – exploratory CountGD++ integration, contracts, examples and regression tests.
- `ULGF_diagnostic/` – ULGF compositing, sequence-generation and validation pipeline.
- `requirements.txt` – Python dependencies for the principal YOLO implementation.

Large datasets, trained model weights, generated videos and full experimental outputs are intentionally excluded from version control. Their preparation procedures and output locations are documented in the notebook.

## Environment

The principal experiment was executed in Google Colab with GPU acceleration. The implementation requires:

- Python 3.10 or later
- PyTorch with CUDA support where available
- Ultralytics YOLO
- OpenCV
- NumPy
- SciPy
- Pandas
- Matplotlib
- TrackEval

Exact runtime versions used by the completed experiment are printed in the submitted notebook.

## Installation

Clone the repository and install the principal project dependencies:

```bash
git clone https://github.com/Martenee/u-traccount.git
cd u-traccount
pip install -r requirements.txt
```
TrackEval is retrieved separately by the notebook from its official repository because it is used directly as an evaluation framework rather than maintained as part of this project.
Reproducing the principal experiment
Open the following notebook in Google Colab:
notebooks/U_TracCount_Submission_Notebook.ipynb

Run the cells from top to bottom. The notebook:
1. verifies the GPU and CUDA environment;
2. mounts Google Drive for persistent experiment outputs;
3. downloads and prepares BrackishMOT;
4. trains and evaluates YOLO11n;
5. performs ByteTrack association;
6. exports predictions in MOTChallenge format;
7. runs TrackEval; and
8. calculates unique-individual counting errors.
If the repository remains private, add a GITHUB_TOKEN entry through Colab Secrets before running the repository-cloning cell. Do not write a token directly into the notebook.
Principal reported results
The held-out detector achieved:
- Precision: 0.7686
- Recall: 0.5116
- mAP@0.5: 0.5506
- mAP@0.5–0.95: 0.2899
- Inference time: approximately 2.19 ms per image
Across the 16 validation sequences, the unique-individual counting evaluation produced:
- MAE: 3.375
- RMSE: 5.0125
- Mean signed error: 2.125
Detailed per-sequence detection, tracking and counting results are retained in the notebook and dissertation.


