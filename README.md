# Satellite Scene Recognition (EuroSAT)

Land-use classification of Sentinel-2 satellite tiles, built up week by week, from hand-crafted
features to CNNs trained from scratch to transfer learning.

**Author:** Abdulloh · Kyungdong University

| Week | Topic | Main file |
|---|---|---|
| 1 | Pretrained ImageNet models: inference and comparison | `week1_pretrained/predict_imagenet.py` |
| 2 | Image gradients, HOG, HOG + colour features with SVMs | `week2_handcrafted/hog_svm.py` |
| 3 | CNN from scratch (`SatNet`), sanity checks, LR sweep, evaluation | `week3_cnn/week3_cnn_eurosat.ipynb` |
| 4 | Transfer learning with 100 labels/class, data efficiency, class imbalance, Grad-CAM | `week4_transfer/week4_transfer_eurosat.ipynb` |

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019): 27,000 RGB tiles, 64×64 px at
10 m/pixel, 10 classes:

`AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · Pasture · PermanentCrop · Residential · River · SeaLake`

It downloads automatically through `torchvision.datasets.EuroSAT` (~90 MB) into `data/`. There is no
official split, so `scenelab.data.stratified_split` makes a 70/15/15 per-class split with seed 42. Weeks 3
and 4 therefore share the same test set.

For scene-level evaluation, classes are grouped into **urban**, **agriculture**, **natural** and **water**.

## Project structure

```
scenelab/              shared code used by every week
  data.py              loading, stratified split, train-only normalisation stats, augmentation
  models.py            SatNet CNN, ResNet-18 builder, receptive field, Grad-CAM
  engine.py            training loop (best-checkpoint restore, frozen-BN handling), prediction
  metrics.py           confusion matrix, per-class scores, top mistakes, group accuracy
  plots.py             learning curves, confusion matrices, confident mistakes
week1_pretrained/      environment check, ImageNet model comparison
week2_handcrafted/     gradients, HOG visualisation, HOG/colour + SVM classifier
week3_cnn/             CNN from scratch notebook
week4_transfer/        transfer learning notebook
samples/               put your own test images here
outputs/               figures saved by the scripts
checkpoints/           model weights saved by the notebooks (git-ignored)
```

## How to run

### Google Colab (recommended for Weeks 3-4)
1. Open the notebook from GitHub in Colab (`File > Open notebook > GitHub`).
2. In the first cell, set `REPO_URL` to this repository's URL.
3. `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all`.

Run Week 3 before Week 4: Week 4 loads `checkpoints/satnet_week3.pt` as a full-data reference.
In Colab both notebooks run inside the same cloned folder, so the checkpoint is found if you run them in the same session.

### Locally
```bash
pip install -r requirements.txt
python week1_pretrained/check_env.py
python week1_pretrained/predict_imagenet.py --image samples/your_photo.jpg
python week2_handcrafted/gradients.py
python week2_handcrafted/hog_visualize.py
python week2_handcrafted/hog_svm.py --per-class 300
jupyter notebook week3_cnn/week3_cnn_eurosat.ipynb
```
Run the scripts from the repository root.

## Results

Fill this in from your own runs.

| Model | Labels used | Test accuracy |
|---|---|---|
| HOG + colour, best SVM (Week 2) | 225 / class | |
| SatNet from scratch (Week 3) | ~1,890 / class | |
| SatNet from scratch (Week 4) | 100 / class | |
| ResNet-18 feature extraction | 100 / class | |
| ResNet-18 fine-tuned | 100 / class | |

## References
- Helber, P. et al. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
- Dalal, N. & Triggs, B. *Histograms of Oriented Gradients for Human Detection.* CVPR, 2005.
- He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
- Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
