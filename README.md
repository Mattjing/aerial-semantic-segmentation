# Aerial Perspective Semantic Segmentation

Semantic segmentation of aerial drone imagery using U-Net with multiple encoder backbones (EfficientNet-B3, MobileNet, ResNet-50, VGG-11). 

## Project Overview

The goal is to classify each pixel of top-down drone images into one of **23 semantic classes** (e.g., paved area, grass, tree, person, car, building, etc.) using fully convolutional encoder-decoder networks.

**Dataset**: [Semantic Drone Dataset](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset) — 400 drone images at 6000×4000 px with pixel-accurate annotations.

## Repository Contents

| File | Description |
|------|-------------|
| `Aerial.ipynb` | Main training & evaluation notebook (U-Net variants) |
| `fpn-efficientnetb3-huuthocse.ipynb` | FPN with EfficientNet-B3 backbone experiment |
| `semantic-segmentation-is-easy-with-pytorch.ipynb` | Reference PyTorch segmentation pipeline |
| `Unet-Efficientnet.pt` | Trained U-Net + EfficientNet-B3 weights |
| `Unet-Mobilenet.pt` | Trained U-Net + MobileNet weights |
| `Unet-resnet50.pt` | Trained U-Net + ResNet-50 weights |
| `Unet-vgg11.pt` | Trained U-Net + VGG-11 weights |
| `archive/class_dict_seg.csv` | Class name ↔ RGB colour mapping (23 classes) |
| `Area segment.png` | Sample segmentation result |
| `mobilenet_test_1.png` | MobileNet inference example |

> **Model weights** are stored via [Git LFS](https://git-lfs.github.com/).

## Setup

### Requirements

```bash
pip install torch torchvision segmentation-models-pytorch
pip install opencv-python pillow numpy pandas matplotlib scikit-learn
pip install tensorflow keras  # for Aerial.ipynb Keras layers
```

Or use the provided notebooks directly in a Jupyter/Kaggle/Colab environment.

### Dataset

Download the dataset from Kaggle and place it at:

```
archive/
  dataset/
    semantic_drone_dataset/
      original_images/
      label_images_semantic/
  class_dict_seg.csv
  RGB_color_image_masks/
```

### Running

Open any notebook in Jupyter and update the `img_path` / `mask_path` variables to point to your local dataset directory, then run all cells.

## Model Results

| Model | Backbone | Weights Size |
|-------|----------|-------------|
| U-Net | EfficientNet-B3 | ~51 MB |
| U-Net | MobileNet | ~26 MB |
| U-Net | ResNet-50 | ~124 MB |
| U-Net | VGG-11 | ~70 MB |

## Dependencies

- Python 3.9+
- PyTorch ≥ 1.12
- `segmentation-models-pytorch`
- TensorFlow / Keras (Aerial.ipynb)
- CUDA 11.2 + cuDNN 8.1 (for GPU training)

## References

- [Semantic Drone Dataset — Kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset)
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- Course: SEP 769 — McMaster University
