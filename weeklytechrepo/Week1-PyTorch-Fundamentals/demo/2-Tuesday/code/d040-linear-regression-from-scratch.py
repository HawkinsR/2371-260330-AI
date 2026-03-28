import torch
import torch.nn as nn  # Contains standard neural network layers (like Linear, Conv2d) and loss functions
import torch.optim as optim  # Contains optimization algorithms (like SGD, Adam) used to update network weights

# 1. Toy Dataset (e.g., predicting salary from years of experience)
# X: Years of Experience [N, 1] where N is number of samples, 1 is the number of features
# y: Salary in thousands [N, 1]
# We use torch.float32 as it is the standard precision for neural network calculations
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)

# The relationship here is roughly y = 20x + 20, which the model will try to learn
y = torch.tensor([[40.0], [60.0], [80.0], [100.0], [120.0]], dtype=torch.float32)

# 2. Define the PyTorch Standard Architecture
# Custom models in PyTorch must inherit from nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        # Always call the superclass __init__ first to register the model internally
        super(LinearRegressionModel, self).__init__()
        
        # nn.Linear represents a fully connected layer (y = wx + b)
        # It automatically initializes random weights (w) and biases (b)!
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # This method defines how the data flows through the network.
        # When you call `model(x)`, it actually executes `model.forward(x)` behind the scenes.
        return self.linear(x)

def train_model():
    print("--- Initializing Linear Regression ---")
    
    # Instantiate the model with 1 input feature (Years of Experience) 
    # and 1 output target (Salary)
    input_dim = 1
    output_dim = 1
    model = LinearRegressionModel(input_dim, output_dim)
    
    # Look at the initial randomly generated weights before training
    print(f"Initial Random Weights:\n{list(model.parameters())}\n")
    
    # 3. Define the Loss Function and Optimizer
    
    # criterion calculates how far off our predictions are from reality
    # nn.MSELoss() (Mean Squared Error) is the standard loss function for regression tasks
    criterion = nn.MSELoss() 
    
    # The optimizer determines HOW to update the model parameters to minimize the loss
    # Adam is an advanced optimizer, typically more robust and faster to converge than vanilla SGD
    # lr=0.1 is the Learning Rate: the step size the optimizer takes when adjusting weights
    optimizer = optim.Adam(model.parameters(), lr=0.1) 
    
    # An epoch is one complete pass through the entire training dataset
    epochs = 100
    
    print("--- Starting Training Loop ---")
    
    # 4. The Training Loop (The standard PyTorch 5-step process)
    for epoch in range(epochs):
        # Step 0: Set model to training mode (important for things like Dropout or BatchNorm)
        model.train() 
        
        # Step 1. Forward Pass: Pass input X through the model to compute predictions
        predictions = model(X)
        
        # Step 2. Compute Loss: Compare predictions against the actual target labels 'y'
        loss = criterion(predictions, y)
        
        # Step 3. Zero the gradients: Crucial step! PyTorch accumulates gradients by default. 
        # We must clear out the previous epoch's gradients before computing new ones.
        optimizer.zero_grad()
        
        # Step 4. Backward Pass: Autograd computes the gradients of the loss with respect 
        # to all model parameters (how exactly to adjust weights to reduce error).
        loss.backward()
        
        # Step 5. Optimization Step: The optimizer uses the gradients to actually update the weights
        optimizer.step()
        
        # Print progress every 1/10th of epochs so we can watch it learn
        if (epoch+1) % (epochs / 10) == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    print("\n--- Training Complete ---")
    print("Final Trained Weights:")
    
    # The true formula was y = 20x + 20. 
    # Notice how close the trained weights are to weight=20, bias=20!
    for name, param in model.named_parameters():
        print(f"  {name}: {param.item():.4f}")
        
    # 5. Inference (Testing the trained model)
    
    # Set model to evaluation mode (turns off Dropout/BatchNorm layers if they exist)
    model.eval() 
    
    # Let's predict the salary for someone with 6 years of experience (which should be ~140k)
    test_input = torch.tensor([[6.0]]) 
    
    # torch.no_grad() disables gradient tracking during inference.
    # This saves memory and significantly speeds up predictions, as we aren't training anymore.
    with torch.no_grad(): 
        prediction = model(test_input)
        
    print(f"\nPrediction for 6 years of experience: {prediction.item():.2f}k")

if __name__ == "__main__":
    train_model()
