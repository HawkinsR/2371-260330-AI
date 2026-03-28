import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class ModerationLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_classes):
        super(ModerationLSTM, self).__init__()
        
        # 1. Initialize the Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        
        # 2. Initialize the LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
                            
        # 3. Initialize the Linear classification head
        self.fc = nn.Linear(hidden_dim, output_classes)

    def forward(self, x):
        # 1. Pass input through the embedding layer
        embedded = self.embedding(x)
        
        # 2. Pass embedded vectors through the LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # 3. Extract the hidden state of the final time step
        final_memory = h_n[-1] # or h_n[0] since it's a single layer
        
        # 4. Pass the final memory through the linear layer
        out = self.fc(final_memory)
        
        return out

def process_chat_logs():
    print("--- Chat Moderation Pipeline ---")
    
    vocab_size = 10
    
    msg1 = torch.tensor([1, 2, 3])          # "You are great" (len 3)
    msg2 = torch.tensor([1, 2, 4, 5])       # "You are terrible idiot" (len 4)
    msg3 = torch.tensor([6])                # "thanks" (len 1)
    msg4 = torch.tensor([1, 2, 4, 1, 2, 5]) # "You are terrible you are idiot" (len 6)
    
    chat_logs = [msg1, msg2, msg3, msg4]
    
    print("Original message lengths:")
    for i, msg in enumerate(chat_logs):
        print(f"  Msg {i+1}: {len(msg)} tokens")
        
    # 1. Pad the sequences using pad_sequence
    padded_batch = pad_sequence(chat_logs, batch_first=True, padding_value=0)
    
    print(f"\nPadded Batch Shape: {padded_batch.shape} (Expected: torch.Size([4, 6]))")
    
    # Initialize the Model
    model = ModerationLSTM(vocab_size=vocab_size, 
                           embedding_dim=16, 
                           hidden_dim=32, 
                           output_classes=2)
                           
    # 2. Pass the padded_batch through the model to get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(padded_batch)
        
    print(f"Output Predictions Shape: {predictions.shape} (Expected: torch.Size([4, 2]))")

if __name__ == "__main__":
    process_chat_logs()
