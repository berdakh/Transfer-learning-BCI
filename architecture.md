# Architecture Documentation

This document provides a detailed explanation of the architecture and methodology used in this subject-independent EEG-based BCI implementation.

## Dataset

The implementation uses the BCI Competition IV Dataset 2a, which includes EEG recordings from 9 subjects performing motor imagery tasks. The dataset has the following characteristics:

- 9 subjects
- 22 EEG channels
- 4 classes of motor imagery: left hand, right hand, feet, tongue
- This implementation focuses on binary classification (left hand vs. right hand)

## Data Processing Pipeline

1. **Loading**: The data is loaded using MNE-Python's interface to the BCI Competition IV Dataset 2a
2. **Epoching**: The continuous EEG data is segmented into epochs around the motor imagery events
3. **Preprocessing**: 
   - Baseline correction
   - Channel-wise standardization (z-score normalization)
   - Class selection (filtering to include only left hand and right hand trials)

## Model Architecture: EEGHybridNet

The `EEGHybridNet` architecture combines elements from EEGNetv4 and EEGConformer to create a powerful model for EEG classification:

### 1. Temporal Convolution Block

- **Purpose**: Capture temporal patterns in the EEG signal
- **Components**:
  - 2D Convolution with kernel size (1, 51) to capture temporal patterns
  - Batch Normalization for training stability
  - ELU activation function
  - Average Pooling to reduce dimensionality
  - Dropout for regularization

### 2. Spatial Convolution Block

- **Purpose**: Learn spatial filters across EEG channels
- **Components**:
  - 2D Convolution with kernel size (n_channels, 1) to capture spatial patterns
  - Batch Normalization
  - ELU activation
  - Average Pooling
  - Dropout

### 3. Separable Convolution Block

- **Purpose**: Efficient feature extraction with reduced parameters
- **Components**:
  - Depthwise Convolution with kernel size (1, 15)
  - Pointwise Convolution with kernel size (1, 1)
  - Batch Normalization
  - ELU activation
  - Average Pooling
  - Dropout

### 4. Self-Attention Block

- **Purpose**: Capture long-range dependencies in the data (inspired by EEGConformer)
- **Components**:
  - Linear layer
  - Layer Normalization
  - GELU activation
  - Linear layer
  - Dropout
  - Residual connection

### 5. Classification Head

- **Purpose**: Final classification of features
- **Components**:
  - Linear layer to reduce dimensionality
  - ELU activation
  - Dropout
  - Linear layer for final classification

## Transfer Learning Methodology

The implementation uses a leave-one-subject-out cross-validation approach with a two-stage fine-tuning strategy:

### 1. Base Model Training

- Train on data from all subjects except the test subject
- Use early stopping based on validation loss
- Save the best model based on validation performance

### 2. Fine-Tuning Stage 1

- Create a new model with the same architecture
- Copy the feature extraction layers from the base model
- Freeze the feature extraction layers
- Train only the classification head on a small portion of the test subject's data
- Use a reduced learning rate (0.1 × base learning rate)

### 3. Fine-Tuning Stage 2

- Unfreeze all layers
- Fine-tune the entire model with an even lower learning rate (0.01 × base learning rate)
- Use early stopping with shorter patience

This approach allows the model to adapt to new subjects while retaining knowledge learned from other subjects, improving cross-subject generalization.

## Evaluation Metrics

The implementation uses the following metrics to evaluate performance:

- **Accuracy**: Proportion of correctly classified trials
- **AUC (Area Under the ROC Curve)**: Measures the model's ability to discriminate between classes
- **Confusion Matrix**: Visualizes the model's performance for each class

## Visualization Components

The implementation includes several visualization components:

1. **EEG Data Visualization**: Displays raw EEG signals for each subject and class
2. **Training History**: Plots training and validation loss over epochs
3. **Confusion Matrices**: Visualizes classification performance for each subject
4. **Overall Results**: Compares performance metrics across subjects for both base and fine-tuned models

These visualizations help in understanding the data, monitoring training progress, and evaluating model performance.
