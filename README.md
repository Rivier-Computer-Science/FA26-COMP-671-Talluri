# Classification of Plant Leaf Diseases Using Deep Learning

COMP-671 (Fall 2026) Deep Learning Final Project, Vardhini Talluri.

Classifies a tomato leaf photo into 10 classes (healthy + 9 diseases) using the PlantVillage dataset. A handcrafted-feature SVM and a small CNN
are compared against ResNet-18 (ImageNet-pretrained), with controlled comparisons for pretraining and data augmentation, plus a failure analysis.
The whole pipeline is in one notebook, [`plant_leaf_disease.ipynb`](plant_leaf_disease.ipynb) (saved outputs included). The written report is
[`report/report.pdf`](report/report.pdf).

## Project structure

```
plant_leaf_disease.ipynb   # full pipeline: data, split, training, evaluation, failure analysis
requirements.txt           # pinned dependencies
data/splits/               # train.csv, val.csv, test.csv (the exact split used)
results/                   # metrics (JSON/CSV), figures, test predictions, saved model weights (.pt)
report/report.pdf          # written report
README.md, LICENSE
```

## Setup

Tested with Python 3.13 on an Apple-silicon Mac (also runs on CUDA or CPU).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

- **Source:** PlantVillage tomato subset (18,155 images, 10 classes) from Hugging Face
  [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage), license CC-BY-SA-3.0 (images are not redistributed here).
- **Download:** automatic. The notebook downloads the ~2.2 GB archive into `data/raw/` on first run and extracts the tomato images.
- **Split (seed 42), grouped by physical leaf so near-duplicate photos never cross splits:**

| Split | Images | How it was made |
|---|---:|---|
| train | 12,442 | official train split minus validation |
| val | 2,074 | ~15% of official train, grouped by leaf, stratified by class |
| test | 3,639 | official PlantVillage test split (tomato only) |

The notebook asserts that no leaf appears in two splits. Known limitation: about 37% of images have no entry in the dataset's leaf-grouping file
(see Sections 2.3 and 6.5 of the report).

## How to run (training and evaluation)

Run the notebook top to bottom. It trains all models, evaluates them on the held-out test set, and produces every table and figure.

```bash
jupyter nbconvert --to notebook --execute plant_leaf_disease.ipynb --inplace
```

(or open the notebook in Jupyter / VS Code and choose "Run All").

- Takes about 1.5 hours on an Apple-silicon laptop the first time, and about 2-3 minutes when the saved results in `results/` are reused.
- A model that already has both a results file and weights in `results/` is loaded instead of retrained. To retrain everything, set
  `FORCE = True` in the first code cell or delete the files in `results/`.
- Quick sanity check (tiny subset, a few minutes, writes to `results_smoke/`):

```bash
SMOKE=1 jupyter nbconvert --to notebook --execute plant_leaf_disease.ipynb --output /tmp/smoke.ipynb
```

## Models and key hyperparameters

| Run | Model | Init | Augmentation |
|---|---|---|---|
| Handcrafted + SVM | HSV/LBP/RGB features, RBF SVM (C chosen on validation) | n/a | n/a |
| `small_cnn_aug` | 4-block CNN, 0.4 M params | random | on |
| `resnet18_scratch_aug` | ResNet-18 | random | on |
| `resnet18_pretrained_noaug` | ResNet-18 | ImageNet | off |
| `resnet18_pretrained_aug` (main) | ResNet-18 | ImageNet | on |

All neural models: 160x160 input, cross-entropy loss, Adam (lr 1e-3, weight decay 1e-4), cosine LR decay, batch size 64, 12 epochs,
best epoch chosen by validation macro-F1, seed 42. All hyperparameters are in the notebook's config cell and in the report's Appendix Table A1.

## Results (test set)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Handcrafted features + SVM | 0.965 | 0.959 |
| Small CNN (scratch) | 0.867 | 0.835 |
| ResNet-18 (scratch) | 0.951 | 0.940 |
| ResNet-18 (pretrained, no aug) | 0.989 | 0.985 |
| **ResNet-18 (pretrained, aug), main** | 0.981 | 0.980 |

One training run per model. Confidence intervals, per-class results, the failure analysis and the discussion are in the report and the notebook.

## Saved weights

Weights for the four trained networks are in `results/*.pt` (the main model is `results/resnet18_pretrained_aug.pt`, 45 MB).
Running the notebook reuses them; deleting them makes the notebook retrain from scratch with the same seed.

## License

See [LICENSE](LICENSE).
