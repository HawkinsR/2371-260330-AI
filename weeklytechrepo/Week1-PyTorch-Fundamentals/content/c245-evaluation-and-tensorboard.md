# Evaluation and TensorBoard

## Learning Objectives

- Explain why accuracy alone is misleading on imbalanced datasets.
- Calculate and interpret Precision, Recall, and F1 Score.
- Read and interpret an AUC-ROC curve and a Confusion Matrix.
- Set up TensorBoard and log scalar metrics from a training loop.
- Interpret a loss curve to identify convergence, overfitting, or divergence.

## Why This Matters

A model with a low training loss is not necessarily a good model. Evaluating performance correctly—especially on imbalanced data—requires metrics that measure what the model is actually getting right and wrong. TensorBoard gives you a live window into training so you can catch problems early, before wasting hours of compute on a model that is heading in the wrong direction.

## The Concept

### Why Accuracy Is Not Enough

**Accuracy** = correct predictions / total predictions. This sounds reasonable, but it fails on imbalanced data. Consider a cancer-screening model where 99% of patients are healthy: a model that always predicts "healthy" achieves 99% accuracy while missing every single cancer case.

Instead, we use metrics that separately account for how the model performs on each class:

- **Precision:** Of all samples the model predicted as positive, what fraction actually were positive? High precision means few false alarms.
- **Recall:** Of all samples that were actually positive, what fraction did the model correctly identify? High recall means few missed detections.
- **F1 Score:** The harmonic mean of Precision and Recall. It is high only when both are high—you cannot game it by sacrificing one for the other.

A **Confusion Matrix** plots True Positives, True Negatives, False Positives, and False Negatives in a table, making it easy to see exactly which types of errors the model is making.

> **Key Term - True/False Positive/Negative:** The four outcomes of a binary classifier. A **True Positive** is a correct positive prediction. A **True Negative** is a correct negative prediction. A **False Positive** (Type I error) is an incorrect positive prediction—a false alarm. A **False Negative** (Type II error) is a missed positive—the model predicted negative when the truth was positive.

> **Key Term - Harmonic Mean:** A type of average that gives more weight to lower values. The F1 Score uses the harmonic mean of Precision and Recall so that a model must perform well on *both* metrics. A model with 100% Precision and 0% Recall will have an F1 of 0.0.

> **Key Term - AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** A metric that measures how well a model separates classes across all possible decision thresholds. A score of 1.0 means perfect separation; a score of 0.5 means the model performs no better than random guessing.

### TensorBoard Logging

**TensorBoard** is a web-based dashboard for tracking training metrics. You write scalar values (like loss and accuracy) from your training loop to a log directory using `SummaryWriter`. Then you launch TensorBoard locally and it renders interactive charts that update in real time.

The loss curve tells you a great deal:
- **Smoothly decreasing:** Training is working.
- **Stopped decreasing:** The model has converged (or is stuck).
- **Diverging:** The learning rate may be too high.
- **Training loss decreasing, validation loss increasing:** The model is overfitting.

> **Key Term - Convergence:** The point during training when the model's loss stops decreasing meaningfully. The model has learned as much as it can given the current data and architecture. Monitoring the loss curve in TensorBoard lets you detect convergence early and avoid wasting compute.

## Code Example

```python
import torch
from torch.utils.tensorboard import SummaryWriter

# SummaryWriter streams log data to the specified directory
writer = SummaryWriter(log_dir='./runs/experiment_1')

for epoch in range(10):
    # Simulated values — in a real loop these would come from your training logic
    train_loss = 0.5 - (0.05 * epoch)
    val_accuracy = 0.6 + (0.03 * epoch)

    # Log scalars: the tag name determines where it appears in TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

# Flush and close the writer when done
writer.close()

# To view the dashboard, run this in a terminal:
# tensorboard --logdir=./runs
```

## Additional Resources

- [Visualizing Models Using TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [Classification Metrics Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
