import torch

def create_and_manipulate_tensor():
    """
    Task 1: Tensor Creation and Device Management
    """
    print("--- Task 1: Tensor Manipulation ---")
    
    # 1. Create a 1D tensor [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # TODO: Initialize the tensor
    my_tensor = None 
    
    # 2. Reshape it into a 2x3 matrix
    # TODO: Use .view() or .reshape()
    reshaped_tensor = None
    
    # 3. Move to GPU if available
    # TODO: Define the device and move the tensor
    
    print(f"Final Tensor:\n{reshaped_tensor}")
    # print(f"Device: {reshaped_tensor.device}") # Uncomment when finished
    
    return reshaped_tensor


def compute_gradients():
    """
    Task 2: Autograd Mechanics
    """
    print("\n--- Task 2: Compute Gradients ---")
    
    # 1. Initialize scalar tensor x = 2.0 with gradient tracking
    # TODO: Create the tensor
    x = None 
    
    # 2. Define the equation: y = 3x^2 + 4x + 2
    # TODO: Write the equation
    y = None
    
    # 3. Compute the backward pass
    # TODO: Call backward on y
    
    # 4. Extract the gradient
    # TODO: Return x.grad.item()
    gradient = None 
    
    print(f"The gradient dy/dx at x=2.0 is: {gradient}")
    return gradient


if __name__ == "__main__":
    create_and_manipulate_tensor()
    compute_gradients()
