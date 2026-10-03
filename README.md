# Classification of Plant Leaf Diseases Using Deep Learning

COMP-671 (Fall 2026) Deep Learning Final Project, Vardhini Talluri.

**Everything lives in one notebook: [`plant_leaf_disease.ipynb`](plant_leaf_disease.ipynb)** (data, models, experiments, failure analysis, discussion). Outputs are saved in it, so you can read the results without re-running.

## Problem

Given a photo of a tomato leaf, predict whether it is healthy or which of nine diseases it has (10 classes).
Early diagnosis helps growers act sooner. **Primary metric: macro-F1** on a held-out test set (classes are imbalanced,
66 to 1,100 test images per class); also accuracy and per-class F1.

## Data and split

PlantVillage tomato subset (18,155 images, 10 classes) from Hugging Face
[`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage) (license CC-BY-SA-3.0; not redistributed here).
The notebook downloads it (~2.2 GB) and builds the split. The exact split is committed in [`data/splits/`](data/splits) (`train.csv`, `val.csv`, `test.csv`).

PlantVillage has several photos per physical leaf, so a random split leaks near-duplicates. We split **by leaf**:

| Split | Images | Source |
|---|---:|---|
| train | 12,442 | official train split minus validation |
| val | 2,074 | ~15% of official train, `StratifiedGroupKFold` by `leaf_id`, stratified by class, seed 42 |
| test | 3,639 | official PlantVillage test split (tomato only) |

The notebook asserts that no leaf is in two splits. The test set is used only to score each finished run; model selection uses validation.
**Limitation:** the dataset's leaf map covers ~63% of images; the rest are treated as their own group, so some near-duplicate leakage is possible
(quantified in the notebook, Fig. 8).

## Models and experiments (course mapping: Prince Ch. 5, 8-11)

| Run | Model | Init | Augmentation | Role |
|---|---|---|---|---|
| Handcrafted + SVM | HSV histograms + LBP texture + RGB stats, RBF SVM | n/a | n/a | non-deep baseline |
| `small_cnn_aug` | 4 x conv-BN-ReLU-pool CNN (0.4 M params) | random | on | deep baseline |
| `resnet18_scratch_aug` | ResNet-18 | random | on | ablation A: pretraining |
| `resnet18_pretrained_noaug` | ResNet-18 | ImageNet | off | ablation B: augmentation |
| `resnet18_pretrained_aug` | ResNet-18 | ImageNet | on | main model |

## Results (test set, single run, seed 42)

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Handcrafted features + SVM | 0.965 | 0.959 |
| Small CNN (scratch, aug) | 0.867 | 0.835 |
| ResNet-18 (scratch, aug) | 0.951 | 0.940 |
| ResNet-18 (pretrained, no aug) | 0.989 | 0.985 |
| **ResNet-18 (pretrained, aug), main** | 0.981 | 0.980 |

Robustness (test macro-F1 under degradation, `results/robustness.csv`): at blur sigma=1 the augmented model keeps 0.935 vs. 0.701 without
augmentation; at 50% darkening 0.975 vs. 0.734. Both models collapse under Gaussian noise (sigma=0.1).

Key findings (details and caveats in the notebook, Section 5):
- **Pretraining is the biggest lever** (+4 macro-F1 points over the same network from scratch).
- **A simple SVM on colour/texture features is a strong baseline** (0.959) and beats both from-scratch deep models.
- **Augmentation costs a little clean accuracy but buys robustness** to blur and darkness; by validation macro-F1 alone the no-augmentation model would be chosen.
- **Hardest errors:** Target Spot called healthy (12/288), Early blight vs. Target Spot; errors are confident (mean 0.77), so confidence does not flag them well.
- **Limits:** lab images only (no field test), one seed, imperfect leaf grouping. Not a diagnostic tool; see the ethics discussion in the notebook.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute plant_leaf_disease.ipynb --inplace   # or open in Jupyter and Run All
```

- Needs ~2.2 GB download and ~1.4 GB disk for the decoded image cache; runs on CUDA, Apple MPS or CPU. On an Apple-silicon laptop the full run took about 1.5 hours.
- Finished runs are cached in `results/` and skipped on re-run; set `FORCE = True` in the setup cell to retrain everything.
- Quick sanity check (about 2 minutes, writes to `results_smoke/`): `SMOKE=1 jupyter nbconvert --to notebook --execute plant_leaf_disease.ipynb --output /tmp/smoke.ipynb`.
- **Weights:** the main model is saved at `results/resnet18_pretrained_aug.pt` (45 MB); the other models' weights are git-ignored and reproduced by re-running the notebook (seed 42).

## Key hyperparameters (all in the notebook's config cell)

Image size 160x160 (ImageNet mean/std normalisation), Adam (lr 1e-3, weight decay 1e-4), cosine LR decay, batch 64, 12 epochs max, early stopping
(patience 4) on validation macro-F1, cross-entropy loss, seed 42. Augmentation: rotation +-20 deg, zoom up to 1.25x, shift +-10%, horizontal/vertical flips,
brightness/contrast jitter +-25%. SVM: RBF kernel, `C` chosen from {1, 10, 100} on validation, balanced class weights.

## Repository contents

```
plant_leaf_disease.ipynb   # the whole project, with saved outputs
requirements.txt           # pinned dependencies
data/splits/               # train/val/test split CSVs
results/                   # metrics (JSON/CSV), figures, test predictions, main model weights
README.md, LICENSE
```
