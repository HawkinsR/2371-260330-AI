# Lab: Image Dataset Pipeline

## The Scenario
Your team needs to train a computer vision model to identify defective parts on an assembly line. However, the raw image data is messy, varies in size, and exceeds the available server RAM. You must architect a PyTorch `Dataset` and `DataLoader` pipeline that efficiently reads images from disk, applies necessary mathematical transformations (resizing and normalizing), and serves the data in manageable mini-batches to the training loop.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e027-dataset_lab.py`.
3. Complete the `DefectDataset` class:
   - Inherit from `Dataset`.
   - Implement `__init__` to store the image paths, labels, and optional transform.
   - Implement `__len__` to return the total number of samples.
   - Implement `__getitem__` to load a simulated image (provided in snippet), apply the transform if it exists, and return the `(image_tensor, label)` tuple.
4. Complete the `build_pipeline` function:
   - Define a `transforms.Compose` pipeline that resizes images to `128x128`, converts them to Tensors, and normalizes them with `mean=[0.5, 0.5, 0.5]` and `std=[0.5, 0.5, 0.5]`.
   - Instantiate your `DefectDataset` with the provided mock paths and your transform.
   - Wrap the dataset in a `DataLoader` with a batch size of `16` and shuffle enabled.
   - Iterate through the dataloader for one simulated epoch and print the shape of the batched images to verify your pipeline.

## Definition of Done
- Your script executes successfully.
- The `Dataset` correctly reports its length.
- The `DataLoader` iteration prints that the batched `images` tensor has the shape `[16, 3, 128, 128]`.
