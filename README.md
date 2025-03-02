# LeNet Implementation for Alzheimer's Disease Classification

## Overview
This project implements a LeNet-5 convolutional neural network for classifying Alzheimer's disease stages from brain MRI images. The solution provides a complete MLOps pipeline for medical image analysis, including data preprocessing, model training, evaluation, and deployment.

## Key Features
- **Adapted LeNet-5 Architecture**: Customized for 64x64 grayscale medical images
- **End-to-End MLOps Pipeline**:
  - S3-based data storage and model artifacts
  - SageMaker training and deployment
  - Automated hyperparameter configuration
  - WandB experiment tracking
- **Medical Imaging Preprocessing**:
  - MRI image normalization
  - Adaptive resizing (64x64 default)
  - Grayscale conversion
- **Model Management**:
  - Multi-class classification (4 stages of Alzheimer's)
  - Model checkpointing every 10 epoch 
  - Comprehensive metrics (Accuracy, Recall, F1-Score)

## Technologies Used
- **Core Framework**: PyTorch 2.2, TorchVision
- **Cloud Infrastructure**: AWS SageMaker, S3
- **Data Processing**: NumPy, PIL, Pandas
- **Experiment Tracking**: Weights & Biases (WandB)

## Dataset
The Alzheimer's MRI dataset contains preprocessed brain images categorized into 4 classes:
The dataset was splitted by 70/15/15

| Class                 | Train | Valid | Test | Total |
|-----------------------|-------|-------|------|-------|
| VeryMildDemented      | 1840  | 394   | 395  | 2629  |
| ModerateDemented      | 1303  | 279   | 280  | 1862  |
| MildDemented          | 1844  | 395   | 396  | 2635  |
| NonDemented           | 1941  | 416   | 417  | 2774  |
|-----------------------|-------|-------|------|-------|
| Total                 | 6928  | 1484  | 1488 | 9900  |


**Preprocessing Pipeline:**
1. Resize to 64x64 pixels
2. Grayscale conversion
3. Intensity normalization
4. S3 storage for distributed training

## Architecture
Modified LeNet-5 structure for medical imaging:

```python
LeNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5))
  (fc1): Linear(16*29*29, 120)  # Input size 128x128
  (fc2): Linear(120, 84)
  (fc3): Linear(84, 4)          # 4 output classes
)
```

**Key Adaptations:**
- Input channels: 1 (grayscale MRI)
- Final FC layer: 4 neurons for disease stages
- Adaptive pooling for variable input sizes
- Batch normalization between layers

## Training Details
**Hyperparameters** (from config.ini):
```ini
learning_rate = 0.0001
batch_size = 16
epochs = 50
resize = 64
```

**MLOps Features:**
- Automatic GPU acceleration
- Model checkpointing to S3
- Training metrics visualization
- Model export as TorchScript

## Results


## Usage
1. **Training:**
```bash
python src/train.py \
  --train s3://cad-brbh-datascience/alzheimer_images/train/ \
  --val s3://cad-brbh-datascience/alzheimer_images/valid/ \
  --num_classes 4 \
  --resize 64
```

2. **Evaluation:**
```bash
python src/evaluate.py \
  --model_path model/checkpoint_20.pth \
  --image_dir data/alzheimer/valid \
  --batch_size 32
```

3. **SageMaker Deployment:**
```python
estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.p3.2xlarge',
    framework_version='2.2',
    hyperparameters={
        'epochs': 50,
        'batch_size': 16,
        'num_classes': 4
    }
)
```

## License
This project is intended for research purposes only. Clinical use requires additional validation and certification.
