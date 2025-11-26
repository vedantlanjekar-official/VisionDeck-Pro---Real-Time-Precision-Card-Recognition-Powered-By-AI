# Model Files

## Important Note

The pre-trained model files (`*.pt`) are not included in this repository due to their large size (~50MB each).

## Available Models

The project uses the following models located in `final_models/`:

- `yolov8m_synthetic.pt` - Main production model (52 classes)
- `yolov8m_tuned.pt` - Fine-tuned model (13 classes - hearts suit)
- `yolov8m.pt` - Base YOLOv8 model

## How to Get Model Files

1. **Download from training**: If you have trained the models, they will be in the `final_models/` directory after training.

2. **Train your own**: Use the training scripts in `model_utils/train.py` with the datasets in `data/` directory.

3. **Contact maintainer**: Reach out to the repository maintainer for access to pre-trained model files.

## Model Requirements

The application requires at least `yolov8m_synthetic.pt` to run the main application. This model file should be placed in the `final_models/` directory.



