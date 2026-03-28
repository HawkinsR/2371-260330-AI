import torch  # The main PyTorch library used for all tensor operations and deep learning functions
import numpy as np  # NumPy is used for traditional numerical computing in Python

def run_tensor_demo():
    print("--- PyTorch vs NumPy ---")
    
    # 1. NumPy Array
    # Create a simple 2D array (matrix) using NumPy
    # This is often how data is initially loaded before deep learning
    np_array = np.array([[1, 2], [3, 4]])
    print(f"NumPy Array:\n{np_array}\n")
    
    # 2. Bridge to PyTorch Tensor
    # Convert the NumPy array into a PyTorch Tensor
    # Tensors are the core data structure in PyTorch, similar to arrays but optimized for deep learning
    tensor = torch.from_numpy(np_array)
    
    # Convert dtype if necessary
    # Neural networks usually expect 32-bit floating point numbers (float32) for standard precision
    tensor = tensor.float() 
    print(f"PyTorch Tensor:\n{tensor}")
    # .shape tells us the dimensions (e.g., 2 rows, 2 columns)
    # .dtype tells us the data type (e.g., float32)
    print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}\n")
    
    # 3. Tensor Operations
    print("--- Tensor Operations ---")
    # Create two 1D tensors (vectors)
    tensor_a = torch.tensor([1.0, 2.0, 3.0])
    tensor_b = torch.tensor([4.0, 5.0, 6.0])
    
    # Element-wise addition
    # PyTorch automatically adds the corresponding elements (1+4, 2+5, 3+6)
    sum_tensor = tensor_a + tensor_b
    print(f"Addition: {sum_tensor}")
    
    # Dot product
    # Computes the dot product: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
    dot_product = torch.dot(tensor_a, tensor_b)
    # .item() extracts the standard Python number from a 1-element tensor
    print(f"Dot Product: {dot_product.item()}\n")
    
    # 4. CPU vs GPU Mechanics
    print("--- GPU Mechanics ---")
    # This is a critical pattern! It checks if an NVIDIA GPU (CUDA) is available.
    # If yes, we use "cuda". If no, we fall back to "cpu".
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected Device: {device}")
    
    # Move tensor to device
    # This sends the data from the computer's main RAM to the GPU's VRAM for fast processing
    tensor_gpu = tensor_a.to(device)
    print(f"Tensor on {device}: {tensor_gpu}")
    print(f"Device attribute: {tensor_gpu.device}\n")
    
    # 5. Autograd Mechanics
    print("--- Autograd Mechanics ---")
    # requires_grad=True tells PyTorch to track all operations performed on this tensor.
    # This enables automatic differentiation (calculating gradients) later for training neural networks.
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([3.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    
    # Forward pass: y = wx + b (The equation of a straight line, common in linear layers)
    y = w * x + b
    print(f"Forward pass output y: {y.item()}")
    
    # Backward pass: Compute gradients (dy/dw, dy/dx, dy/db)
    # This tells PyTorch to look at the 'y' result and calculate how much 'w', 'x', and 'b' 
    # contributed to that result using calculus (chain rule).
    y.backward()
    
    # Now we print the calculated gradients.
    # The gradient of (w*x+b) with respect to w is x. So w.grad = 2.0
    print(f"Gradient dy/dw (should be x=2.0): {w.grad.item()}")
    # The gradient of (w*x+b) with respect to x is w. So x.grad = 3.0
    print(f"Gradient dy/dx (should be w=3.0): {x.grad.item()}")
    # The gradient of (w*x+b) with respect to b is 1. So b.grad = 1.0
    print(f"Gradient dy/db (should be 1.0): {b.grad.item()}")

if __name__ == "__main__":
    run_tensor_demo()
