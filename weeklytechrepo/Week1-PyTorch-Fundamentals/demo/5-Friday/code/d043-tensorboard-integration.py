import torch
import torch.nn as nn
import torch.optim as optim
# SummaryWriter is the core engine that writes out log files for the TensorBoard UI to read
from torch.utils.tensorboard import SummaryWriter 

# 1. Toy Architecture (A very simple Multi-Layer Perceptron)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5) # Input layer (10 features) to hidden layer (5 neurons)
        self.fc2 = nn.Linear(5, 1)  # Hidden layer to output (1 prediction)
        self.sigmoid = nn.Sigmoid() # Sigmoid squashes the output to a probability between [0, 1]

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Apply ReLU activation after the first layer
        return self.sigmoid(self.fc2(x))

def run_tensorboard_demo():
    print("--- Initializing TensorBoard SummaryWriter ---")
    model = SimpleNet()
    
    # Binary Cross Entropy Loss is used when the output is a single probability (0 to 1)
    criterion = nn.BCELoss() 
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 2. Instantiate the Writer
    # By default, logs save to a folder like './runs/Mar10_12-34-56_hostname'
    # Defining a specific explicitly named directory 'runs/demo_experiment_1' keeps things organized
    writer = SummaryWriter(log_dir='runs/demo_experiment_1')
    
    # Optional but highly recommended: Log the model graph itself to TensorBoard!
    # To do this, TensorBoard needs to run a 'dummy' input through the network to trace the execution path
    dummy_input = torch.randn(1, 10)
    writer.add_graph(model, dummy_input)
    print("- Model graph logged to TensorBoard.")

    print("\n--- Starting Training Loop with Logging ---")
    epochs = 100
    for epoch in range(epochs):
        # Fake data generation per epoch to simulate batches of data
        inputs = torch.randn(16, 10)
        labels = torch.randint(0, 2, (16, 1)).float()
        
        # Standard Training Steps (Zero gradients, Forward pass, Loss, Backward pass, Update weights)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Fake validation loss (just for visual plotting in demo, we pretend validation is slightly worse)
        val_loss = loss.item() * 1.2
        
        # 3. Log to TensorBoard Using writer.add_scalar
        # Syntax: writer.add_scalar('Chart_Name', Y_Value, X_Value)
        # We group charts by using a slash, e.g., 'Loss/Train' and 'Loss/Validation' 
        # will show up in the same 'Loss' chart section in the UI!
        writer.add_scalar('Loss/Train', loss.item(), epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Calculate a fake "Accuracy" for demonstration purposes
        # If output probability > 0.5, predict Class 1, else Class 0
        predictions = (outputs > 0.5).float() 
        # Calculate what percentage of predictions matches the actual labels
        accuracy = (predictions == labels).sum().item() / len(labels)
        
        # Log the accuracy scalar
        writer.add_scalar('Metrics/Accuracy', accuracy, epoch)

        if epoch % 20 == 0:
            print(f"Logged Epoch {epoch}: Train Loss {loss.item():.4f} | Acc {accuracy:.2f}")

    # 4. Flush and Close the DB connection!
    # CRITICAL: Always flush and close the writer to ensure all data is written to disk from memory
    writer.flush()
    writer.close()
    
    print("\n--- Training Complete ---")
    print("Run `tensorboard --logdir=runs` in your terminal to view the dashboard in your browser!")

if __name__ == "__main__":
    run_tensorboard_demo()
