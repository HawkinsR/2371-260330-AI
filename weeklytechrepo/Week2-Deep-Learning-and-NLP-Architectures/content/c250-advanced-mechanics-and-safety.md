# Advanced Mechanics and Safety

## Learning Objectives

- Formalize robust Reproducibility patterns locking random seeds across hardware layers.
- Apply GPU acceleration techniques evaluating Mixed Precision Training (AMP).
- Stabilize erratic learning via mathematical Gradient Clipping logic.
- Prevent extensive Overfitting implementing Early Stopping Logic.
- Architect dynamic Training Callbacks to manage long-running training loops efficiently.
- Evaluate model logic extracting Explainability with SHAP values.
- Redact Sensitive PII (Personally Identifiable Information) using Microsoft Presidio.
- Compress prompt contexts using LLM Lingua for efficient token usage and inference.
- Synthesize an Algorithm selection framework deciding between distinct model classes.

## Why This Matters

Building a neural network is only the first 20% of an AI engineer's job. The remaining 80% is orchestrating training runs that don't crash, don't overfit, run fast enough on limited GPU clusters, and reliably generate the same result twice. Mastering these mechanics transitions you from a student writing Python scripts to a professional engineering reliable deep learning pipelines.

## The Concept

### Reproducibility and Early Stopping

Stochastic Gradient Descent (SGD) is inherently random, as is dropout and weight initialization. If we don't lock our random seeds (in Python, NumPy, and PyTorch), our model will learn differently every time we restart the script.
Furthermore, letting a model train for 100 epochs blindly is dangerous. Early Stopping monitors the validation loss. If the validation loss stops improving (or starts getting worse) for a set number of epochs (the "patience"), the training loop automatically halts, saving the best checkpoint. This guarantees we never overfit.

> **Key Term - Random Seed:** A starting number given to a random number generator to make its output predictable and repeatable. Setting `torch.manual_seed(42)` means every random operation (weight initialization, dropout, data shuffling) will produce the same result every time, making experiments reproducible and comparable.

> **Key Term - Early Stopping:** An automatic training termination technique. It monitors the validation loss after every epoch. If the loss doesn't improve for a set number of epochs (the "patience" window), training halts and the best weights found so far are restored. This prevents wasted compute and prevents overfitting.

```python
import torch

def train_with_early_stopping(model, optimizer, criterion, train_loader, val_loader, patience=5):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_weights = None

    for epoch in range(100):  # max 100 epochs
        # --- Training phase ---
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_loss += criterion(model(inputs), targets).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()  # Save best weights
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                model.load_state_dict(best_weights)   # Restore best weights
                break

    return model
```

### Training Callback Architecture

A **callback** is a reusable object that plugs into predefined events in the training loop (e.g., `on_epoch_end`, `on_batch_end`). Rather than writing ad-hoc logic directly into your loop, you define callbacks separately and attach them. This keeps the main training loop clean and makes logic reusable across experiments.

Common callback responsibilities:
- **Early Stopping** — halt training on plateau (implemented above)
- **Model Checkpointing** — save `state_dict` only at new best accuracy
- **Metric Logging** — write scalars to TensorBoard or a CSV

```python
class ModelCheckpoint:
    """Saves the model state_dict when validation loss improves."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.best_loss = float('inf')

    def on_epoch_end(self, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.filepath)
            print(f"  Checkpoint saved to {self.filepath} (val_loss={val_loss:.4f})")

# Usage:
checkpoint_cb = ModelCheckpoint('best_model.pth')
for epoch in range(50):
    # ... training ...
    val_loss = evaluate(model, val_loader)
    checkpoint_cb.on_epoch_end(model, val_loss)
```

> **Key Term - Callback Architecture:** A design pattern where training loop events trigger pluggable functions or class methods. This pattern separates *what happens* (train a batch) from *what to do about it* (log a metric, save a file, adjust a learning rate), enabling modular and reusable training orchestration.

### Mixed Precision and Gradient Clipping

Standard tensors are 32-bit floats (`float32`). GPUs have dedicated hardware (Tensor Cores) that execute 16-bit math (`float16`) blisteringly fast. Automatic Mixed Precision (AMP) dynamically scales the math so that non-critical operations run in 16-bit, cutting memory usage in half and doubling training speed without losing accuracy.

Occasionally, exploding gradients will cause our loss to hit `NaN` (Not a Number). Gradient Clipping enforces a strict maximum ceiling on gradient sizes just before `optimizer.step()`, preventing parameters from updating radically in a single batch.

> **Key Term - Mixed Precision Training (FP16/FP32):** Using both 16-bit (half-precision) and 32-bit (full-precision) floating-point numbers during training. The forward pass and gradient calculations run in fast FP16, while sensitive operations (like the optimizer update) run in stable FP32. This reduces GPU memory usage and speeds up training significantly on modern hardware.

> **Key Term - Gradient Clipping:** A technique that caps the magnitude (size) of gradients before the optimizer update. If gradients grow too large ("exploding gradients"), they can cause wild weight updates that destabilize training. Clipping enforces a max gradient norm (e.g., `max_norm=1.0`), ensuring updates are always bounded and stable.

### Explainability with SHAP

Deep learning models are notoriously "black boxes." SHAP (SHapley Additive exPlanations) is a game-theoretic diagnostic tool. It systematically calculates exactly how much each specific input feature (like a pixel in an image or a word in a sentence) contributed to the final probability output, rendering visual heatmaps explaining the model's logic.

> **Key Term - Black Box Model:** A model whose internal reasoning is not human-interpretable. We can observe the inputs and outputs, but we cannot easily explain *why* the model made a specific decision. Deep neural networks are often black boxes, which poses challenges in regulated industries (healthcare, finance) where decisions must be explainable.

> **Key Term - SHAP (SHapley Additive exPlanations):** An explainability technique that assigns each input feature a "contribution score" to the model's final prediction. A positive SHAP value means that feature pushed the prediction higher; a negative value means it pushed it lower. This turns the black box into a transparent model with auditable reasoning.

```python
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
```

### Algorithm Selection Framework

A critical skill for AI engineers is knowing *which architecture to reach for* when starting a new project. Use this framework as a decision guide:

| Task Type | Input | Recommended Architecture | Why |
|---|---|---|---|
| Image classification, object detection, segmentation | Images, video frames | **CNN** | Preserves spatial structure; parameter-efficient via filter sharing |
| Sequence prediction, time-series forecasting | Short-to-medium sequential data | **RNN / LSTM / GRU** | Built-in temporal memory; efficient for moderate sequence lengths |
| NLP classification, NER, Q&A | Text (any length) | **BERT (Encoder Transformer)** | Bidirectional context; state-of-the-art on language understanding tasks |
| Text generation, code completion, chatbots | Text generation | **GPT (Decoder Transformer)** | Autoregressive generation; scales to billions of parameters |
| Translation, summarization | Text-to-text tasks | **T5 / Encoder-Decoder Transformer** | Full seq2seq architecture; adapts to any text-in, text-out task |
| Tabular data, structured features | CSV/database rows | **Gradient Boosting (XGBoost) or MLP** | Deep learning rarely outperforms tree-based methods on structured data |

> **Key Term - Algorithm Selection:** The process of choosing the appropriate model architecture based on the data modality (images, sequences, tables), task type (classification, generation, regression), and practical constraints (dataset size, latency, explainability requirements). No single architecture wins across all tasks.

### AI Safety: PII Masking and Context Compression

As models move into production, safety and cost-efficiency become paramount.

1.  **PII Masking (Microsoft Presidio):**
    Training models on raw customer data is a major security risk. **PII Masking** identifies and redacts Personally Identifiable Information (Names, SSNs, Credit Card numbers) before the data ever touches a model. This ensures compliance with regulations like GDPR and HIPAA.
2.  **LLMLingua (Context Compression):**
    Modern LLMs have massive context windows, but feeding 100,000 tokens into an API is expensive and slow. **LLMLingua** uses a small, fast model to identify "noise" in a prompt and remove it, compressing the context by up to 20x while retaining the original instruction's meaning.

> **Key Term - PII (Personally Identifiable Information):** Any data that could be used to identify a specific individual. Masking this data is a core requirement for responsible AI engineering.

> **Key Term - Prompt Compression:** The process of removing redundant or irrelevant tokens from a prompt to reduce latency and API costs without degrading the model's performance.

## Code Example

```python
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler  # PyTorch 2.x unified AMP API

# 1. Reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# 2. Device setup and Gradient Scaler for AMP
# AMP provides the most benefit on CUDA hardware, but the pattern works on CPU too
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"

scaler = GradScaler(device_type)
model = nn.Linear(10, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy Data
inputs = torch.randn(32, 10).to(device)
targets = torch.randint(0, 2, (32,)).to(device)
criterion = nn.CrossEntropyLoss()

# 3. Training Loop with AMP and Clipping
optimizer.zero_grad()

# Forward pass in half-precision (float16)
with autocast(device_type):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Backward pass scales the loss to preserve tiny gradients
scaler.scale(loss).backward()

# Unscale the gradients and perform Gradient Clipping
# Max norm set to 1.0 to prevent exploding gradients
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Step optimizer and update scale for next iteration
scaler.step(optimizer)
scaler.update()
```

```python
# --- PII Masking with Microsoft Presidio ---
# pip install presidio-analyzer presidio-anonymizer
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = "My name is John Smith and my email is john.smith@example.com"

# Detect PII entities in the text
results = analyzer.analyze(text=text, language="en")

# Replace detected PII with placeholder tags
anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
print(anonymized.text)
# Output: "My name is <PERSON> and my email is <EMAIL_ADDRESS>"
```

```python
# --- Prompt Compression with LLMLingua ---
# pip install llmlingua
from llmlingua import PromptCompressor

compressor = PromptCompressor()

long_prompt = """
You are an expert assistant. Given the following detailed context about the history
of artificial intelligence, neural networks, deep learning frameworks, and the
evolution of transformer architectures from early attention mechanisms through
modern multi-head self-attention, please answer the user's question concisely.

Question: What is self-attention?
"""

compressed = compressor.compress_prompt(long_prompt, rate=0.5)
print(f"Original tokens : {compressed['origin_tokens']}")
print(f"Compressed tokens: {compressed['compressed_tokens']}")
print(f"Compressed prompt : {compressed['compressed_prompt']}")
```

## Additional Resources

- [PyTorch Automatic Mixed Precision Examples](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [SHAP GitHub Repository and Documentation](https://github.com/slundberg/shap)
- [Microsoft Presidio — PII Detection and Anonymization](https://microsoft.github.io/presidio/)
