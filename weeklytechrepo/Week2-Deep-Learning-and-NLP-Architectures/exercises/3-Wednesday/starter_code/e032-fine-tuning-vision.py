import torch
import torch.nn as nn
import torchvision.models as models

def build_transfer_model(num_classes=5):
    """
    Task 1: Load, Freeze, and Modify ResNet
    """
    print("--- Building Transfer Learning Model ---")
    
    # 1. Load pretrained ResNet18
    # TODO: Load ResNet18 with default weights
    model = None
    
    # 2. Freeze the backbone
    # TODO: Loop through model.parameters() and set requires_grad to False
    
    
    # 3. Replace the classification head
    # TODO: Get the number of in_features from the current model.fc
    in_features = 0
    
    # TODO: Replace model.fc with a new nn.Linear layer outputting 'num_classes'
    
    
    return model

def verify_gradients(model):
    """
    Task 2: Verify Only the Head is Training
    """
    print("\n--- Verifying Trainable Parameters ---")
    trainable_count = 0
    
    # TODO: Loop through model.named_parameters()
    # Check if param.requires_grad is True. 
    # If it is, increment trainable_count and print the name of the parameter.
    
    
    print(f"Total trainable parameter tensors: {trainable_count} (Expected: 2)")
    return trainable_count

if __name__ == "__main__":
    # Build the model
    bird_model = build_transfer_model(num_classes=5)
    
    # Verify the gradients
    if bird_model is not None:
        verify_gradients(bird_model)
        
        # Test a forward pass
        print("\n--- Testing Forward Pass ---")
        dummy_input = torch.randn(4, 3, 224, 224)
        bird_model.eval() # Always set to eval when testing
        output = bird_model(dummy_input)
        print(f"Output shape: {output.shape} (Expected: torch.Size([4, 5]))")
