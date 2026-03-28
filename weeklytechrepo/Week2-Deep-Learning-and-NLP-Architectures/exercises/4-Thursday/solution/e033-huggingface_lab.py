import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def initialize_nlp_pipeline(model_name):
    print(f"--- Initializing {model_name} Pipeline ---")
    
    # 1. Load the AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Load the AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return tokenizer, model

def analyze_sentiment(tokenizer, model, reviews):
    print("\n--- Analyzing Reviews ---")
    
    # 1. Tokenize the reviews
    inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")
    
    # 2. Set the model to evaluation mode
    model.eval()
    
    # 3. Perform the forward pass without tracking gradients
    with torch.no_grad():
        outputs = model(**inputs)
        
    print(f"Raw Logits Shape: {outputs.logits.shape}")
    
    # 4. Convert the raw logits to probabilities using Softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 5. Get the predicted class indices using argmax
    predicted_classes = torch.argmax(probs, dim=-1)
    
    # Evaluate and print results
    labels = model.config.id2label
    
    print("\nResults:")
    for i, review in enumerate(reviews):
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
