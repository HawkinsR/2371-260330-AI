import shap
import torch
import torch.nn as nn
import numpy as np

# A simple trained model (in practice, your actual trained model)
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
model.eval()

# Background data for SHAP (a small sample of training data)
background = torch.randn(100, 10)
test_input = torch.randn(5, 10)

# DeepExplainer works directly with PyTorch models
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_input)

# Visualize the impact on the first prediction class (Class 0)
# Note: Newer SHAP versions return shape (batch, features, classes)
shap.summary_plot(
    shap_values[:, :, 0],
    test_input.numpy(),
    feature_names=[f'feature_{i}' for i in range(10)]
)