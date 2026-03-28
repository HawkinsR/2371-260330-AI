# Lab: Tensor Manipulation and Autograd

## The Scenario
You are developing a physics engine simulation and need to leverage GPU acceleration to compute the trajectory of thousands of particles simultaneously. To do this, you must construct the basic building blocks in PyTorch using Tensors. You will also need to calculate the gradient (rate of change) of a specific function to simulate velocity.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e025-tensor_lab.py`.
3. Complete the `create_and_manipulate_tensor` function:
   - Create a 1D PyTorch tensor containing the numbers 1 through 6 (inclusive) as floats.
   - Reshape this 1D tensor into a 2x3 matrix.
   - Move the matrix to the GPU if CUDA is available, otherwise keep it on the CPU.
4. Complete the `compute_gradients` function:
   - Initialize a scalar tensor `x` with the value `2.0` and tell PyTorch to track its gradients.
   - Define the function $y = 3x^2 + 4x + 2$.
   - Execute the backward pass to compute the gradient of `y` with respect to `x`.
   - Return the calculated gradient.
   
## Definition of Done
- Your script runs without errors.
- The 2x3 matrix prints correctly and its `.device` attribute reflects the optimal available hardware.
- The printed gradient for $y$ at $x=2.0$ accurately matches the analytical derivative ($dy/dx = 6x + 4$, so $dy/dx$ at $x=2.0$ should be $16.0$).
