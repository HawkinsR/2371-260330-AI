import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

class ModerationLSTM(nn.Module):
    """
    Task 1: Build the LSTM Architecture
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_classes):
        super(ModerationLSTM, self).__init__()
        
        # 1. TODO: Initialize the Embedding layer (remember padding_idx=0)
        self.embedding = None 
        
        # 2. TODO: Initialize the LSTM (remember batch_first=True)
        self.lstm = None
        
        # 3. TODO: Initialize the Linear classification head
        self.fc = None

    def forward(self, x):
        # 1. TODO: Pass input through the embedding layer
        embedded = None
        
        # 2. TODO: Pass embedded vectors through the LSTM
        # Remember it returns: output, (h_n, c_n)
        
        # 3. TODO: Extract the hidden state of the final time step
        # Extract from h_n. Assuming a single layer, unidirectional LSTM, 
        # h_n shape is [1, batch_size, hidden_dim]. Get the 0th index.
        final_memory = None
        
        # 4. TODO: Pass the final memory through the linear layer
        out = None
        
        return out

def process_chat_logs():
    """
    Task 2: Pad Sequences and Predict
    """
    print("--- Chat Moderation Pipeline ---")
    
    # Vocabulary mapping (Simulated)
    # 0=PAD, 1=You, 2=are, 3=great, 4=terrible, 5=idiot, 6=thanks
    vocab_size = 10
    
    # 4 distinct chat messages of variable length
    msg1 = torch.tensor([1, 2, 3])          # "You are great" (len 3)
    msg2 = torch.tensor([1, 2, 4, 5])       # "You are terrible idiot" (len 4)
    msg3 = torch.tensor([6])                # "thanks" (len 1)
    msg4 = torch.tensor([1, 2, 4, 1, 2, 5]) # "You are terrible you are idiot" (len 6)
    
    # Our batch of sequences
    chat_logs = [msg1, msg2, msg3, msg4]
    
    print("Original message lengths:")
    for i, msg in enumerate(chat_logs):
        print(f"  Msg {i+1}: {len(msg)} tokens")
        
    # 1. TODO: Pad the sequences using pad_sequence
    # Remember: batch_first=True, padding_value=0
    padded_batch = None
    
    if padded_batch is not None:
        print(f"\nPadded Batch Shape: {padded_batch.shape} (Expected: torch.Size([4, 6]))")
    
    # Initialize the Model
    model = ModerationLSTM(vocab_size=vocab_size, 
                           embedding_dim=16, 
                           hidden_dim=32, 
                           output_classes=2)
                           
    # 2. TODO: Pass the padded_batch through the model to get predictions
    model.eval()
    with torch.no_grad():
        predictions = None
        
    if predictions is not None:
        print(f"Output Predictions Shape: {predictions.shape} (Expected: torch.Size([4, 2]))")

def log_training_to_tensorboard():
    """
    Task 3: Log a Simulated Training Run to TensorBoard
    """
    print("\n--- TensorBoard Logging ---")
    
    # TODO: Create a SummaryWriter that writes logs to 'runs/moderation_lstm'
    writer = None
    
    if writer is not None:
        # TODO: Simulate 5 training epochs.
        # For each epoch (0 through 4), calculate a decaying loss: loss = 1.0 / (epoch + 1)
        # Log it with: writer.add_scalar('Loss/train', loss, epoch)
        
        
        # TODO: Close the writer when done.
        
        
        log_dir = 'runs/moderation_lstm'
        print(f"TensorBoard logs saved to: {log_dir}")
        print(f"  Run: tensorboard --logdir={log_dir} to visualize")

if __name__ == "__main__":
    process_chat_logs()
    log_training_to_tensorboard()
