"""
Demo: LSTM Sequence Prediction
This script demonstrates how to pad variable length sequences, pass them 
iteratively through an embedding layer, and process them with an LSTM cell 
for classification.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence # Utility to standardize sentence lengths

class TextClassifierLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifierLSTM, self).__init__()
        
        # 1. Embedding Layer: maps integer IDs (words) to dense vectors of numbers
        # This allows the network to learn geometric relationships between words (e.g. King - Man + Woman = Queen)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim,
                                      padding_idx=0) # Index 0 represents <PAD>, which the network will ignore 
                                      
        # 2. LSTM (Long Short-Term Memory) Layer
        # LSTMs are special RNNs that can remember context over long sequences without 'forgetting'
        self.lstm = nn.LSTM(input_size=embedding_dim, # Size of the incoming word vector
                            hidden_size=hidden_dim,   # Size of the persistent 'memory' vector
                            batch_first=True)         # Tells PyTorch input shape is [Batch, Sequence, Features]
                            
        # 3. Output Layer: Maps the final memory state to our target classes (e.g. Positive/Negative)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, lengths):
        # x is a batch of sentences, already padded to the same length. Shape: [batch_size, max_seq_length]
        
        # 1. Convert Word IDs to embedded thick vectors
        # embedded shape: [batch_size, max_seq_length, embedding_dim]
        embedded = self.embedding(x)
        
        # 2. Pass sequential data through the LSTM
        # The LSTM processes the sentence word-by-word, updating a hidden memory state inside
        # lstm_out contains the memory state at EVERY step of the sequence
        # hn is the final hidden state (the "summary" of the whole sentence)
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # Extract the hidden state from the final time step
        # Since this is a simple unidirectional 1-layer LSTM, the final state is at index 0.
        # final_hidden shape: [batch_size, hidden_dim] tensor
        final_hidden = hn[0]
        
        # 3. Pass the final sentence summary vector to the linear classifier
        # output shape: [batch_size, num_classes]
        output = self.fc(final_hidden)
        
        return output

def demonstrate_lstm_pipeline():
    print("--- LSTM Sequence Processing Pipeline ---")
    
    # 1. Create dummy vocabulary and text sequences of variable lengths
    # Real NLP pipelines use tokenizers (like HuggingFace's) to map words to these numbers
    # Let's say: 0=PAD, 1=I, 2=love, 3=PyTorch, 4=hate, 5=bugs
    vocab_size = 10
    
    # Sequence 1: "I love PyTorch" (length 3)
    seq1 = torch.tensor([1, 2, 3])
    # Sequence 2: "I hate bugs" (length 3)
    seq2 = torch.tensor([1, 4, 5])
    # Sequence 3: "PyTorch" (length 1)
    seq3 = torch.tensor([3])
    
    # Pack them into a list
    sequences = [seq1, seq2, seq3]
    # We must record their real lengths before we pad them, so the network knows where the real sentence ends
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    print("Original Variable Length Sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Seq {i+1} (len {len(seq)}): {seq.tolist()}")
        
    # 2. Pad the sequences to the length of the longest sequence in the batch
    # PyTorch requires rectangular tensors for batch processing. We add 0s to make them identical length.
    print("\nPadding sequences...")
    # pad_sequence expects a list of tensors
    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    print("Padded Batch Tensor:")
    print(padded_batch)
    print(f"Padded Batch Shape: {padded_batch.shape} -> [batch_size, max_seq_len]")
    
    # 3. Initialize Model Architecture
    embedding_dim = 16 # Each word will become a 16-number vector
    hidden_dim = 32    # The LSTM memory state will be a 32-number vector
    num_classes = 2    # e.g., Positive / Negative sentiment
    
    model = TextClassifierLSTM(vocab_size, embedding_dim, hidden_dim, num_classes)
    
    # Check the embedding representation of the padded batch
    print("\nPassing through Embedding Layer...")
    embeddings = model.embedding(padded_batch)
    print(f"Embeddings Shape: {embeddings.shape} -> [batch_size, max_seq_len, embedding_dim]")
    
    # 4. Perform Forward Pass
    print("\nPerforming Forward Pass through LSTM and Classifier...")
    model.eval()
    
    # We use torch.no_grad() because we are simulating an inference/prediction phase, not training
    with torch.no_grad():
        predictions = model(padded_batch, lengths)
        
    print(f"Predictions Shape: {predictions.shape} -> [batch_size, num_classes]")
    print("Raw Logits Output:")
    print(predictions)
    
    print("-" * 50)

if __name__ == "__main__":
    demonstrate_lstm_pipeline()
