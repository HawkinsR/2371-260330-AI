import torch
import torch.nn as nn
import torchvision.models as models

def build_transfer_model(num_classes=5):
    print("--- Building Transfer Learning Model ---")
    
    # 1. Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Replace the classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def verify_gradients(model):
    print("\n--- Verifying Trainable Parameters ---")
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
            print(f"Trainable: {name}")
            
    print(f"Total trainable parameter tensors: {trainable_count} (Expected: 2)")
    return trainable_count

if __name__ == "__main__":
    # Build the model
    bird_model = build_transfer_model(num_classes=5)
    
    # Verify the gradients
    verify_gradients(bird_model)
    
    # Test a forward pass
    print("\n--- Testing Forward Pass ---")
    dummy_input = torch.randn(4, 3, 224, 224)
    bird_model.eval()
    with torch.no_grad():
        output = bird_model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: torch.Size([4, 5]))")
