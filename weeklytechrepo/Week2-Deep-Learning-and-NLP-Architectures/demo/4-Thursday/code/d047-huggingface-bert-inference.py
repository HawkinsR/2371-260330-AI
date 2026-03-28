"""
Demo: Hugging Face BERT Inference
This script demonstrates how to tokenize text, perform inference with pre-trained 
BERT, and extract context embeddings using the Hugging Face transformers library.
"""

import torch
# Hugging Face 'transformers' library is the industry standard for NLP models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

def demonstrate_huggingface_pipeline():
    print("--- Hugging Face BERT Pipeline ---")
    
    # Define a lightweight pre-trained model for fast inference
    # DistilBERT is smaller, faster, and cheaper than standard BERT
    # "finetuned-sst-2" means it already knows how to classify text as Positive/Negative Sentiment
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading '{model_name}' from Hugging Face Hub...")
    
    # 1. Initialize Tokenizer
    # The tokenizer must perfectly match the specific BERT variant being used
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Initialize Model for Classification
    # AutoModelForSequenceClassification automatically adds a linear layer (head) on top of BERT
    model_cls = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # 3. Preparation: Raw Text to Tensors
    text = "Transformers completely revolutionized the AI industry! We love learning about them."
    print(f"\nOriginal Text: '{text}'")
    
    # The tokenizer handles splitting words into subwords (tokens), mapping them to integer IDs,
    # and creating the attention mask (which tells the model what is real text vs padding).
    # return_tensors="pt" tells it to return PyTorch tensors instead of standard Python lists
    inputs = tokenizer(text, return_tensors="pt")
    
    print("\nTokenized Inputs:")
    # These are the actual numbers fed into the Neural Network
    print(f"Input IDs (Vocabulary indices): {inputs['input_ids']}")
    print(f"Attention Mask: {inputs['attention_mask']}")
    
    # We can reverse the tokenization to see what the model actually sees:
    # Notice special tokens like [CLS] (beginning of sentence) and [SEP] (end of sentence)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Tokens Breakdown: {tokens}")
    
    # 4. Inference Phase
    print("\nPerforming Inference...")
    model_cls.eval() # Disable dropout layers
    
    with torch.no_grad(): # Disable gradient tracking to save memory and increase speed
        # The **inputs syntax automatically unpacks our dictionary (input_ids and attention_mask)
        outputs = model_cls(**inputs)
        
    print(f"Output Logits: {outputs.logits}")
    
    # Convert raw logits to probabilities (0.0 to 1.0) using Softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Find the index of the highest probability (0 or 1)
    predicted_class_id = torch.argmax(probs, dim=-1).item()
    
    # Map the predicted ID back to the human-readable label
    # SST-2 is a sentiment dataset: 0 = Negative, 1 = Positive
    labels = model_cls.config.id2label
    prediction_label = labels[predicted_class_id]
    
    print(f"\nFinal Prediction: {prediction_label} (Confidence: {probs[0][predicted_class_id]:.4f})")
    print("-" * 50)
    
    # 5. BONUS: Extracting Context Embeddings (The hidden states)
    print("--- Extracting Context Embeddings ('hidden states') ---")
    print("Loading base AutoModel (no classification head)...")
    
    # Load the base model without the sentiment classification head at the end
    base_model = AutoModel.from_pretrained("distilbert-base-uncased")
    base_model.eval()
    
    with torch.no_grad():
        # Pass the same text inputs we generated earlier
        base_outputs = base_model(**inputs)
        
    # The last_hidden_state represents the rich contextual embeddings for every single token
    embeddings = base_outputs.last_hidden_state
    
    print(f"Embeddings Tensor Shape: {embeddings.shape} -> [batch_size, sequence_length, hidden_dimension]")
    print("Notice the hidden dimension (typically 768 for BERT-base architectures).")
    print("These dense vectors contain the 'meaning' of the sentence mathematically encoded via Self-Attention.")
    print("-" * 50)

if __name__ == "__main__":
    demonstrate_huggingface_pipeline()
