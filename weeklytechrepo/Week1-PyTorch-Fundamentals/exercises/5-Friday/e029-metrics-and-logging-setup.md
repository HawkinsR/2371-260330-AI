# Lab: Metrics and TensorBoard Logging

## The Scenario

Your classification model from Thursday is training, but the stakeholders are complaining that "Accuracy" alone doesn't reflect real-world performance on an imbalanced dataset. Furthermore, watching a terminal printout scroll by is no longer acceptable for enterprise monitoring. You need to upgrade the training loop to calculate business-relevant metrics (Precision, Recall, F1 Score) and log them all to a live TensorBoard dashboard.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e029-metrics_lab.py`.
3. Complete the TensorBoard setup:
   - Initialize a `SummaryWriter` pointing to `runs/metrics_lab`.
   - In the training loop, log the training loss using `.add_scalar('Loss/Train', ...)`.
4. Complete the Validation Phase:
   - After running predictions on the validation set, convert outputs to binary class labels (threshold at 0.5).
   - Use `sklearn.metrics.classification_report` to print Precision, Recall, and F1 Score to the terminal.
   - Extract the weighted F1 Score using `sklearn.metrics.f1_score` with `average='weighted'`.
   - Log the validation loss and F1 Score to TensorBoard using `.add_scalar()`.
5. Finalize the run by flushing and closing the `SummaryWriter`.

## Definition of Done

- The script runs successfully and prints a `classification_report` to the terminal for each validation epoch.
- A `runs/metrics_lab` directory is generated containing TensorBoard event files.
- Inspecting the TensorBoard UI (`tensorboard --logdir=runs`) shows clear line graphs for Train Loss, Validation Loss, and Validation F1 Score.
