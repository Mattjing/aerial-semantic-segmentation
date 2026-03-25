# Aerial Perspective Semantic Segmentation

Semantic segmentation of aerial drone imagery using U-Net with multiple encoder backbones (EfficientNet-B3, MobileNet, ResNet-50, VGG-11). 

## Project Overview

The goal is to classify each pixel of top-down drone images into one of **23 semantic classes** (e.g., paved area, grass, tree, person, car, building, etc.) using fully convolutional encoder-decoder networks. The system is then applied to real-world Google Maps aerial imagery to demonstrate practical applicability in urban planning and analysis.

**Dataset**: [Semantic Drone Dataset](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset) — 400 drone images at 6000×4000 px (captured from 5–30 m altitude) with pixel-accurate annotations across 23 categories.

**Authors**: Haocun Jing, Shu Li, Jingcheng Liu, Shiyu Chen — McMaster University, SEP 769

## Key Features

- **Multi-model comparison**: FPN (EfficientNet-B3 backbone) vs. U-Net with four backbones (MobileNet, EfficientNet-B3, ResNet-50, VGG-11)
- **Real-world validation**: Segmentation applied to large-area Google Maps nadir-view imagery to assess practical performance
- **High-risk / dangerous zone identification**: Post-processing step that detects when cars and pedestrians are in close proximity. Binary masks are created for both object classes, contours are extracted via OpenCV, and areas where the car–pedestrian distance falls below a defined proximity threshold are highlighted as high-risk zones on the output image

## Repository Structure

```
aerial-semantic-segmentation/
├── scripts/                  # Training & inference Python scripts
│   ├── Aerial.py             # Main U-Net training & evaluation script
│   ├── fpn-efficientnetb3-huuthocse.py       # FPN + EfficientNet-B3 experiment
│   └── semantic-segmentation-is-easy-with-pytorch.py  # Reference pipeline
├── models/                   # Trained model weights (stored via Git LFS)
│   ├── Unet-Efficientnet.pt
│   ├── Unet-Mobilenet.pt
│   ├── Unet-resnet50.pt
│   └── Unet-vgg11.pt
├── data/
│   └── class_dict_seg.csv    # Class name ↔ RGB colour mapping (23 classes)
├── results/                  # Sample segmentation output images
│   ├── Area segment.png
│   └── mobilenet_test_1.png
└── README.md
```

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

Open any script in `scripts/` and update the `img_path` / `mask_path` variables to point to your local dataset directory, then run it directly or inside a Jupyter environment.

## Model Results

| Model | Backbone | Precision | Recall | mIoU | Weights |
|-------|----------|-----------|--------|------|---------|
| FPN | EfficientNet-B3 | **90%** | **88%** | **0.85** | `models/Unet-Efficientnet.pt` |
| U-Net | MobileNet | 85% | 82% | 0.78 | `models/Unet-Mobilenet.pt` |
| U-Net | ResNet-50 | — | — | — | `models/Unet-resnet50.pt` |
| U-Net | VGG-11 | — | — | — | `models/Unet-vgg11.pt` |

**Key findings:**
- The FPN model excels at detecting both small and large objects thanks to its multi-scale feature pyramid — best overall accuracy.
- U-Net with MobileNet offers efficient feature extraction and strong performance on large objects (buildings, trees) but struggles with smaller objects (pedestrians, vehicles).
- Both top-performing models (FPN-EfficientNet-B3 and U-Net-MobileNet) were validated on a real-world Google Maps aerial image, confirming practical applicability.

## High-Risk Zone Detection

After segmentation, a post-processing pipeline identifies **dangerous zones** where vehicles and pedestrians are in close proximity:

1. **Mask creation** — binary masks are extracted for the `car` and `person` classes using their RGB colour values from `data/class_dict_seg.csv`
2. **Contour detection** — OpenCV finds the outlines of each detected object in the masks
3. **Proximity analysis** — distances between car and pedestrian contours are computed; pairs below a configurable distance threshold are flagged
4. **Visualisation** — high-risk areas are highlighted on the segmented image in a distinct colour for easy identification

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
