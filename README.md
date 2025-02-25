# LeNet Implementation for Fashion MNIST Classification

## Overview
This project demonstrates a PyTorch implementation of the LeNet convolutional neural network for image classification on the Fashion MNIST dataset. The implementation serves as a technical showcase for computer vision workflows, ML engineering practices, and MLOps fundamentals.

## Key Features
- **LeNet-5 Architecture**: Classic CNN implementation with modern PyTorch conventions
- **Fashion MNIST Dataset**: Classifies 10 categories of fashion items from 28x28 grayscale images
- **MLOps Pipeline**:
  - Training with validation metrics (accuracy, recall, F1-score)
  - Model checkpointing and saving
  - Metrics visualization (loss/accuracy curves)
  - Inference pipeline with results export
- **Production-ready Features**:
  - Dataset abstraction with custom DataLoader
  - GPU acceleration support
  - Model serialization/deserialization
  - Comprehensive image preprocessing

## Technologies Used
- **Core ML**: PyTorch, TorchVision, Sagemaker, s3
- **Data Handling**: pandas, numpy
- **MLOps**: Model checkpointing, metrics tracking, hyperparameter configuration



## Results

=== Evaluation Metrics ===
Epochs: 10
Learning-rate: 0.001
Accuracy: 0.8318

Classification Report:
              precision    recall  f1-score   support

  ankle boot       0.93      0.93      0.93        98
         bag       0.97      0.94      0.95        98
        coat       0.72      0.76      0.74        98
       dress       0.67      0.93      0.78        98
    pullover       0.73      0.78      0.75        98
      sandal       0.94      0.94      0.94        98
       shirt       0.64      0.48      0.55        98
     sneaker       0.90      0.93      0.92        99
     trouser       0.99      1.00      0.99        98
  tshirt_top       0.86      0.64      0.74        98

    accuracy                           0.83       981
   macro avg       0.84      0.83      0.83       981
weighted avg       0.84      0.83      0.83       981


Confusion Matrix:
[[91  0  0  0  0  2  0  5  0  0]
 [ 0 92  0  2  1  1  0  1  1  0]
 [ 0  0 74  9  7  0  8  0  0  0]
 [ 0  0  6 91  1  0  0  0  0  0]
 [ 0  1  8  2 76  0 10  0  0  1]
 [ 3  0  0  0  0 92  0  3  0  0]
 [ 0  2 15 10 15  0 47  0  0  9]
 [ 4  0  0  0  0  3  0 92  0  0]
 [ 0  0  0  0  0  0  0  0 98  0]
 [ 0  0  0 22  4  0  8  1  0 63]]


=== Evaluation Metrics ===
Epochs: 50
Learning-rate: 0.001
Accuracy: 0.8522

Classification Report:
              precision    recall  f1-score   support

  ankle boot       0.98      0.91      0.94        98
         bag       0.95      0.93      0.94        98
        coat       0.73      0.76      0.74        98
       dress       0.75      0.87      0.80        98
    pullover       0.83      0.81      0.82        98
      sandal       0.92      0.96      0.94        98
       shirt       0.69      0.62      0.65        98
     sneaker       0.90      0.94      0.92        99
     trouser       0.96      0.99      0.97        98
  tshirt_top       0.83      0.74      0.78        98

    accuracy                           0.85       981
   macro avg       0.85      0.85      0.85       981
weighted avg       0.85      0.85      0.85       981


Confusion Matrix:
[[89  0  0  0  0  2  0  7  0  0]
 [ 0 91  1  3  0  1  2  0  0  0]
 [ 0  3 74  8  6  0  7  0  0  0]
 [ 0  0  7 85  2  0  2  0  2  0]
 [ 0  0  9  3 79  0  6  0  0  1]
 [ 1  0  0  0  0 94  0  3  0  0]
 [ 0  2 10  3  6  1 61  0  1 14]
 [ 1  0  0  0  0  4  1 93  0  0]
 [ 0  0  0  1  0  0  0  0 97  0]
 [ 0  0  1 11  2  0 10  0  1 73]]




=== Evaluation Metrics ===
Epochs: 100
Learning-rate: 0.001
Accuracy: 0.8573

Classification Report:
              precision    recall  f1-score   support

  ankle boot       0.96      0.94      0.95        98
         bag       0.94      0.93      0.93        98
        coat       0.74      0.84      0.78        98
       dress       0.81      0.85      0.83        98
    pullover       0.82      0.78      0.80        98
      sandal       0.93      0.95      0.94        98
       shirt       0.67      0.54      0.60        98
     sneaker       0.91      0.92      0.91        99
     trouser       0.94      0.99      0.97        98
  tshirt_top       0.83      0.85      0.84        98

    accuracy                           0.86       981
   macro avg       0.85      0.86      0.85       981
weighted avg       0.85      0.86      0.85       981






## Usage
### Training
```bash
python src/train.py \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --num_classes 10
```

### Evaluation
```bash
python src/evaluate.py \
    --image_dir path/to/images \
    --model_path path/to/checkpoint.pth \
    --save_dir results/
```

## Development Setup
```bash
pip install -r requirements
```

## Acknowledgments
- Original LeNet paper: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- Fashion MNIST dataset: [Repository](https://github.com/zalandoresearch/fashion-mnist)

