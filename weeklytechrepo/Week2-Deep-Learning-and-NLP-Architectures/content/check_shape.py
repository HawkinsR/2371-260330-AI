import shap
import torch
import torch.nn as nn
import numpy as np

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
model.eval()

background = torch.randn(100, 10)
test_input = torch.randn(5, 10)

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_input)

print("Type of shap_values:", type(shap_values))
if isinstance(shap_values, list):
    print("List length:", len(shap_values))
    print("Shape of element 0:", shap_values[0].shape)
else:
    print("Shape:", shap_values.shape)
