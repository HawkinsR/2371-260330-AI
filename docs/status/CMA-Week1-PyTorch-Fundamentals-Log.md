# Weekly Epic: Establish the foundational understanding of PyTorch, including tensor operations, neural network construction, and the complete training lifecycle

## 1-Monday

### Written Content

- [x] Create `c240b-intro-to-aiml-fundamentals.md`: Foundational AI/ML Concepts, Machine Learning Lifecycle, Dataset Splitting, Regression vs Classification, Loss Functions, Tensor Math, Neural Network Anatomy, Activation Functions, Forward/Backward Propagation, and Evaluation Metrics.
- [x] Create `c241-pytorch-tensors-and-autograd.md`: Introduction to the ML Lifecycle, Supervised vs Unsupervised learning, PyTorch vs NumPy: Tensors & Bridges, Tensor Shapes, dtypes, and Operations, GPU vs CPU: `.to(device)` & CUDA, Autograd Mechanics & Computational Graphs.

### Instructor Demo

- [x] Create `d039-tensor-operations-and-gpu.py`: Demonstrate tensor mechanics, moving shapes to GPU, and basic autograd graphs.

### Trainee Exercise

- [x] Create `e025-tensor-manipulation.md`: Guide trainees through creating tensors, performing mathematical operations, and executing backpropagation manually.

## 2-Tuesday

### Written Content

- [x] Create `c242-linear-models-and-optimizers.md`: Linear Regression in PyTorch, The `nn.Module` Class Structure, The `forward()` method & Implementations, Loss Functions: MSE vs CrossEntropy, Optimizers: `SGD` & `Adam` Nuances, Learning Rate Schedulers & Decay.

### Instructor Demo

- [x] Create `d040-linear-regression-from-scratch.py`: Show how to build a basic `nn.Module` regression model and optimize it using MSE and Adam/SGD.

### Trainee Exercise

- [x] Create `e026-custom-nn-module-training.md`: Have trainees build their own simple linear and cross-entropy classification models and implement learning rate decay.

## 3-Wednesday

### Written Content

- [x] Create `c243-dataloaders-and-logistic-regression.md`: Logistic Regression Implementation, Handling Class Imbalance, `DataLoader` & Batching Strategies, TorchVision Datasets & Transforms, Custom Dataset Classes (`__getitem__`), Image Normalization & Standardization.

### Instructor Demo

- [x] Create `d041-custom-dataset-and-transforms.py`: Demonstrate creating a custom dataset class and fetching items with TorchVision transforms and a DataLoader.

### Trainee Exercise

- [x] Create `e027-image-dataset-pipeline.md`: Construct a complete data pipeline reading custom images and normalizing them using batch processing.

## 4-Thursday

### Written Content

- [x] Create `c244-mlps-and-training-loops.md`: Multi-Layer Perceptrons (MLP) intuition, The Perceptron and Activation Functions, Hidden Layers, Width vs Depth, Backpropagation Intuition, Gradient Descent and Loss Functions, Overfitting, Underfitting, Bias-Variance, Regularization concepts (L1, L2, Dropout), The Training Loop (Step-by-step), The Validation Loop & Checkpointing.

### Instructor Demo

- [x] Create `d042-complete-mlp-training-loop.py`: Walk through building a multi-layer perceptron, applying dropout, and coding out a complete training and validation loop with checkpointing.

### Trainee Exercise

- [x] Create `e028-end-to-end-mlp-classification.md`: Build a structured MLP, write a robust training/validation loop, and save the best model weights.

## 5-Friday

### Written Content

- [x] Create `c245-evaluation-and-tensorboard.md`: Evaluation & Monitoring, Evaluation metrics: Precision, Recall, F1, AUC-ROC and Confusion Matrix, TensorBoard Setup & Integration, Logging Metrics & Visualizing Graphs.

### Instructor Demo

- [x] Create `d043-tensorboard-integration.py`: Integrate TensorBoard into existing loops to visualize scalars, loss curves, and evaluation metrics dynamically.

### Trainee Exercise

- [x] Create `e029-metrics-and-logging-setup.md`: Calculate and log F1/AUC-ROC directly into TensorBoard and plot confusion matrices from a validation dataset.
