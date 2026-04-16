# Lab: Fine-Tuning Pre-Trained Models

## The Scenario
Your company wants to build a mobile app to identify 5 different species of local birds. You don't have millions of bird images or weeks of GPU time to train a CNN from scratch. Instead, you will use Transfer Learning. You will download a ResNet18 model that has already learned to extract powerful image features from the massive ImageNet dataset. Your task is to freeze those learned features so they don't get destroyed, and only train a brand-new classification head on your specific 5 bird species.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e032-fine-tuning-vision.py`.

3. Complete the `build_transfer_model` function:
   - Load a pre-trained `resnet18` model from `torchvision.models`. Ensure you are using the default pre-trained weights.
   - Iterate through every parameter in the model and set `requires_grad = False`. This "freezes" the convolutional backbone.
   - Replace the final fully-connected layer (`model.fc`) with a new `nn.Linear` layer. The input features should match the original `fc` layer's input features, and the output should be `5` (for our 5 bird species).
4. Complete the `verify_gradients` function:
   - Loop through the named parameters of your modified model.
   - Count how many parameters have `requires_grad == True`. If you froze the backbone and replaced the head correctly, only the weights and biases of your *new* `fc` layer should be trainable (Total = 2).
   
## Definition of Done
- The script executes without errors.
- The `verify_gradients` function prints exactly: `Trainable parameters: 2`.
- You simulate a forward pass with a dummy tensor of shape `[4, 3, 224, 224]` and it outputs a tensor of shape `[4, 5]`.
