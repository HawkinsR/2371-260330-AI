# Lab: CNN Image Classification

## The Scenario
You are building an image classification pipeline for a veterinary clinic to automatically sort intake photos into "Dogs" or "Cats". Because standard linear networks lose the 2D spatial arrangement of pixels, you must construct a Convolutional Neural Network (CNN) that slides filters over the images to detect edges, textures, and eventually complex shapes like ears and paws. 

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e030-cnn_lab.py`.
3. Complete the `PetClassifierCNN` class:
   - Inherit from `nn.Module`.
   - **Layer 1:** Define a 2D Convolution (`nn.Conv2d`) that takes 3 input channels (RGB) and outputs 16 feature maps using a 3x3 kernel and a padding of 1.
   - **Layer 2:** Define a Max Pooling layer (`nn.MaxPool2d`) with a 2x2 kernel and a stride of 2 to cut the spatial dimensions in half.
   - **Layer 3:** Define a second 2D Convolution that takes the 16 channels and outputs 32 feature maps (3x3 kernel, padding 1).
   - **Layer 4:** Define a Fully Connected output layer (`nn.Linear`). *Hint: You will need to calculate the flattened dimension. The input image is 64x64. After two pooling layers, what is the spatial size? Multiply that by the 32 channels.*
   - **Forward Pass:** Route the input through Conv1 -> ReLU -> Pool -> Conv2 -> ReLU -> Pool -> Flatten -> Linear.
4. Complete the `test_forward_pass` function:
   - Instantiate your model.
   - Pass the dummy `[Batch=4, Channels=3, Height=64, Width=64]` image batch through the model.
   - Print the shape of the final output.

## Definition of Done
- The script executes without dimension mismatch errors.
- The printed output shape is `[4, 2]`, representing 4 images and 2 class prediction logits (Dog or Cat).
