import torch

def create_and_manipulate_tensor():
    """
    Task 1: Tensor Creation and Device Management
    """
    print("--- Task 1: Tensor Manipulation ---")
    
    # 1. Create a 1D tensor [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    my_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # 2. Reshape it into a 2x3 matrix
    reshaped_tensor = my_tensor.view(2, 3)
    
    # 3. Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reshaped_tensor = reshaped_tensor.to(device)
    
    print(f"Final Tensor:\n{reshaped_tensor}")
    print(f"Device: {reshaped_tensor.device}")
    
    return reshaped_tensor


def compute_gradients():
    """
    Task 2: Autograd Mechanics
    """
    print("\n--- Task 2: Compute Gradients ---")
    
    # 1. Initialize scalar tensor x = 2.0 with gradient tracking
    x = torch.tensor([2.0], requires_grad=True)
    
    # 2. Define the equation: y = 3x^2 + 4x + 2
    y = 3 * (x ** 2) + 4 * x + 2
    
    # 3. Compute the backward pass
    y.backward()
    
    # 4. Extract the gradient
    gradient = x.grad.item()
    
    print(f"The gradient dy/dx at x=2.0 is: {gradient}")
    return gradient


if __name__ == "__main__":
    create_and_manipulate_tensor()
    compute_gradients()
