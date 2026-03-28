import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def initialize_nlp_pipeline(model_name):
    """
    Task 1: Initialize the Tokenizer and Model
    """
    print(f"--- Initializing {model_name} Pipeline ---")
    
    # 1. TODO: Load the AutoTokenizer
    tokenizer = None
    
    # 2. TODO: Load the AutoModelForSequenceClassification
    model = None
    
    return tokenizer, model

def analyze_sentiment(tokenizer, model, reviews):
    """
    Task 2: Tokenize and Perform Inference
    """
    print("\n--- Analyzing Reviews ---")
    
    # 1. TODO: Tokenize the reviews
    # Remember to set padding=True, truncation=True, and return_tensors="pt"
    inputs = None
    
    # 2. TODO: Set the model to evaluation mode
    
    
    # 3. TODO: Perform the forward pass without tracking gradients
    
    outputs = None # Replace this with the model's output
        
    print(f"Raw Logits Shape: {outputs.logits.shape}")
    
    # 4. TODO: Convert the raw logits to probabilities using Softmax
    probs = None
    
    # 5. TODO: Get the predicted class indices using argmax
    predicted_classes = None
    
    # Evaluate and print results
    labels = model.config.id2label # Dictionary mapping {0: 'NEGATIVE', 1: 'POSITIVE'}
    
    print("\nResults:")
    for i, review in enumerate(reviews):
        # Extract the label and confidence for this specific review
        class_id = predicted_classes[i].item()
        label = labels[class_id]
        confidence = probs[i][class_id].item()
        
        print(f"Review: '{review}'")
        print(f"  -> Prediction: {label} (Confidence: {confidence:.4f})\n")

if __name__ == "__main__":
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    
    my_tokenizer, my_model = initialize_nlp_pipeline(MODEL_NAME)
    
    customer_reviews = [
        "This product exceeded all my expectations. Absolutely fantastic!",
        "It broke after just two days. Terrible build quality.",
        "The shipping was slightly delayed, but the item itself is pretty good."
    ]
    
    if my_tokenizer and my_model:
        analyze_sentiment(my_tokenizer, my_model, customer_reviews)
