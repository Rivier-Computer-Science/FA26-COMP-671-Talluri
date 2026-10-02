# Classification of Plant Leaf Diseases Using Deep Learning

COMP-671 (Fall 2026) Deep Learning Final Project, Vardhini Talluri.

> **Status:** Work in progress. Sections marked _(planned)_ describe the intended design and will be updated as each part is implemented and verified.

## Problem

Given a photo of a single plant leaf, predict whether the plant is healthy or which disease it has. Timely identification of crop disease helps farmers act early and reduce losses. The project is limited to one crop and a small number of classes so that it can be completed and evaluated rigorously within a semester.

- **Input:** RGB image of a leaf (resized to 224x224).
- **Output:** One class label (healthy or a specific disease).
- **Primary metric:** Macro-F1 on a held-out test set (classes are imbalanced).
- **Secondary metrics:** Accuracy, per-class precision/recall/F1, confusion matrix.

## Approach _(planned)_

Course alignment (Prince): loss functions and fitting (Ch. 5-6), performance measurement (Ch. 8), regularization and augmentation (Ch. 9), CNNs (Ch. 10), residual networks and batch normalization (Ch. 11).

| Role | Model |
|------|-------|
| Non-deep baseline | Color-histogram and texture features + logistic regression / SVM |
| Deep baseline | Small CNN trained from scratch |
| Main model | ResNet-18 (transfer learning from ImageNet) |

### Controlled comparisons

1. Data augmentation on vs. off (same model, same seed).
2. Pretrained vs. random initialization (same ResNet-18).
3. Optional: SGD vs. Adam.

### Failure analysis

Most confused class pairs, Grad-CAM inspection (does the model attend to lesions or to the background?), and robustness to background change and blur.

## Data _(planned)_

- **Source:** PlantVillage dataset, restricted to a single crop (e.g. tomato).
- **Split:** Fixed, seeded, stratified 70/15/15 train/validation/test split. Split files will be saved in `data/splits/` so results can be reproduced exactly.
- **Download:** Instructions will be added to `data/README.md`. Raw data is not committed to the repository.
- **Known limitation:** PlantVillage images are taken against uniform backgrounds, so accuracy here likely overstates performance on real field images.

## Repository structure _(planned)_

```
.
├── README.md
├── requirements.txt        # pinned dependencies
├── configs/                # hyperparameters per experiment
├── data/                   # data instructions and split files (no raw images)
├── src/                    # data loading, models, training, evaluation
├── scripts/                # entry points for train / eval / inference
├── results/                # metrics, figures, saved weights or links
└── report/                 # written report
```

## Setup _(planned)_

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage _(planned)_

Commands will be finalized once the scripts exist.

```bash
python scripts/prepare_data.py --config configs/base.yaml
python scripts/train.py --config configs/resnet18_pretrained.yaml
python scripts/evaluate.py --config configs/resnet18_pretrained.yaml
```

## Results

To be added after experiments are run.

## Ethics and responsible use

To be expanded in the report. Key concerns: dataset bias toward lab-style images, the risk of farmers over-trusting a wrong prediction, and compute cost of training. The model is a decision aid and does not replace expert diagnosis.

## License

See [LICENSE](LICENSE).
