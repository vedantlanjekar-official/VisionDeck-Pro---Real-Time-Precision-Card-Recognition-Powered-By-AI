# Datasets

## Important Note

The dataset files (images and labels) are not included in this repository due to their large size (20,000+ images, several GB).

## Available Datasets

The project includes the following datasets in `data/`:

1. **Synthetic Dataset** (`data/synthetic_dataset/`)
   - 20,000 synthetic card images
   - 52 classes (all standard playing cards)
   - Format: YOLOv8

2. **Real Dataset** (`data/real_dataset/`)
   - 100 real-world images
   - 13 classes (hearts suit)
   - Format: YOLOv8

3. **Real Augmented Dataset** (`data/real_augmented_dataset/`)
   - 1,000 augmented images
   - 13 classes (hearts suit)
   - Format: YOLOv8

4. **Combined Dataset** (`data/combined_dataset/`)
   - 1,100 images (combined real + synthetic)
   - 13 classes
   - Format: YOLOv8

## Dataset Structure

Each dataset follows YOLOv8 format:

```
dataset_name/
├── train/
│   ├── images/    # Training images (.jpg)
│   └── labels/    # YOLO annotations (.txt)
├── valid/
│   ├── images/    # Validation images
│   └── labels/    # YOLO annotations
├── test/
│   ├── images/    # Test images
│   └── labels/    # YOLO annotations
└── data.yaml      # Dataset configuration
```

## How to Get Datasets

1. **Kaggle**: The synthetic dataset is available on Kaggle - Playing Cards Object Detection Dataset
2. **Train your own**: Create your own dataset following the YOLOv8 format
3. **Contact maintainer**: Reach out to the repository maintainer for dataset access

## Dataset Sources

- **Synthetic Dataset**: Kaggle (CC0 Public Domain License)
- **Real Dataset**: Custom collection by the project maintainer



