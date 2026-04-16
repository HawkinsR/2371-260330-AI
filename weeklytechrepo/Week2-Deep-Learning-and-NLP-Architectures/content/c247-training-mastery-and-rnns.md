# Training Mastery and RNNs

## Learning Objectives

- Understand why standard feedforward networks cannot process sequential data like text.
- Map discrete words to continuous numeric vectors using `nn.Embedding` layers.
- Explain the internal feedback loop that defines Recurrent Neural Networks (RNNs).
- Describe how LSTMs and GRUs solve the vanishing gradient problem in long sequences.
- Configure Learning Rate Schedulers to optimize the training process.
- Integrate TensorBoard for real-time metric visualization and graph logging.
- Implement Training Callbacks for automated logic like early stopping.

## Why This Matters

Images have fixed dimensions (e.g., 224×224 pixels), but text does not. A sentence can be 5 words or 500. A standard feedforward network processes its entire input at once with no sense of order—the word "not" appearing before "good" versus after it produces the same output, which is clearly wrong for understanding meaning.

Recurrent Neural Networks process sequences one step at a time, maintaining an internal hidden state that accumulates context as each word is read. This makes RNNs the foundational architecture for Natural Language Processing (NLP).

However, building a model is only half the battle. **Training Mastery**—the ability to monitor loss curves, schedule learning rates, and automate training logic—is what separates experimental code from production-ready systems. Tools like TensorBoard and strategies like LR Decay ensure that your models converge faster and more reliably.

## The Concept

### Word Embeddings

Machine learning models work with numbers, not strings. The simplest approach is to assign a unique integer to each word (e.g., "apple" = 1, "banana" = 2). The problem is that integers imply a mathematical relationship that does not exist—"banana" is not twice "apple."

An `nn.Embedding` layer solves this by mapping each word integer to a dense floating-point vector of a fixed size (e.g., 64 numbers). These vectors are learned during training. Words that appear in similar contexts end up with similar vectors. The word "king" and "queen" might be geometrically close in the embedding space, while "king" and "keyboard" are far apart.

The key advantage is that the model learns *meaningful relationships between words* entirely from the data, without any human-defined rules.

> **Key Term - Word Embedding:** A learned, dense numeric vector that represents a single word. Each dimension of the vector captures some aspect of the word's meaning. Embeddings are learned during training so that semantically similar words end up near each other in the vector space.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =============================================================================
# Change this text to see how the embeddings adapt!
# The model will learn which words appear in similar contexts.
# Try substituting words, or adding new sentences.
# =============================================================================
TEXT = """
the king is a strong man .
the queen is a wise woman .
the boy is a young man .
the girl is a young woman .
the prince is a young king .
the princess is a young queen .
"""

# --- 1. Data Preparation ---
# Tokenize and build vocabulary
words = TEXT.lower().split()
vocab = list(set(words))
word_to_ix = {w: i for i, w in enumerate(vocab)}

# Generate context-target pairs (CBOW: predict center word from context)
# We use a window of 1 word before and 1 word after
CONTEXT_SIZE = 1
data = []
for i in range(CONTEXT_SIZE, len(words) - CONTEXT_SIZE):
    context = [words[i - 1], words[i + 1]]
    target = words[i]
    data.append((context, target))

# --- 2. Model Definition ---
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        # The embedding layer maps integer indices to dense vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # Get embeddings for context words and average them
        embeds = self.embeddings(inputs).mean(dim=0).unsqueeze(0)
        # Predict the probability of each word in the vocabulary being the target
        out = self.linear(embeds)
        return out

# --- 3. Training Loop ---
# We use 2 dimensions so we can graph them directly!
EMBEDDING_DIM = 2 
model = CBOW(len(vocab), EMBEDDING_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)

print("Training embedding model on the text...")
for epoch in range(150):
    total_loss = 0
    for context, target in data:
        # Prepare inputs and targets as PyTorch tensors
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)

        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, target_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    if (epoch + 1) % 30 == 0:
         print(f"Epoch {epoch+1:3d} | Loss: {total_loss:.4f}")

# --- 4. Visualizing the Embeddings ---
# Extract the learned embedding matrix
trained_embeddings = model.embeddings.weight.data.clone()

plt.figure(figsize=(10, 8))
# Plot each word's embedding
for word, i in word_to_ix.items():
    x, y = trained_embeddings[i].numpy()
    plt.scatter(x, y, color='blue')
    plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points')

plt.title("Learned Word Embeddings (2D Space)")
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

### RNNs and the Hidden State

An RNN processes a sequence one token at a time. At each step, it takes two inputs: the current word embedding and the **hidden state** from the previous step. It produces a new hidden state that is passed forward to the next step. At the end of the sequence, the final hidden state represents a summary of everything the network has read.

This feedback loop is what makes RNNs "recurrent." The hidden state is the network's working memory.

> **Key Term - Hidden State:** A vector produced at each time step of an RNN that encodes the model's accumulated understanding of the sequence so far. It acts as the RNN's short-term memory, carrying information from earlier in the sequence into later processing steps.

### LSTMs and GRUs

Basic RNNs suffer from the vanishing gradient problem: gradients shrink exponentially as they are backpropagated through many time steps, so the model fails to retain context from early in a long sequence.

**Long Short-Term Memory (LSTM)** cells solve this by adding a separate **cell state** alongside the hidden state. Gate mechanisms (input, forget, output gates) control what information is written to, retained in, or erased from the cell state. This allows the model to carry relevant context across hundreds of time steps without gradients vanishing.

**Gated Recurrent Units (GRUs)** are a simplified version of LSTMs with fewer parameters and faster training. They are often preferred when the training dataset is smaller.

> **Key Term - LSTM (Long Short-Term Memory):** A type of RNN cell that uses gating mechanisms to control information flow through a dedicated cell state. The forget gate removes irrelevant past information, the input gate adds new information, and the output gate determines what part of the cell state to expose as the hidden state. LSTMs can learn long-range dependencies in sequences.

> **Key Term - Variable-Length Sequences:** In a mini-batch, sentences are rarely the same length. To process them efficiently, shorter sequences are padded with a special token to match the longest sequence in the batch. PyTorch's `pack_padded_sequence` utility tells the LSTM to ignore the padding during computation.

### Learning Rate Schedulers

A fixed learning rate is rarely optimal for the entire training run. You typically want a high learning rate at the start to make big progress, and a smaller one at the end to "fine-tune" the weights into the local minimum.

PyTorch provides `torch.optim.lr_scheduler` for this. Common strategies include:
- **StepLR:** Decays the learning rate by a multiplicative factor (`gamma`) every `step_size` epochs. Simple and predictable.
- **ReduceLROnPlateau:** Monitors a metric (like validation loss) and reduces the LR automatically when that metric stops improving. More adaptive for uncertain training runs.

> **Key Term - Learning Rate Decay:** The systematic reduction of the learning rate over the course of training. A large initial LR makes fast early progress; a small later LR allows precise convergence into a good local minimum without overshooting it.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

model = nn.Linear(10,2)
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# Multiply LR by 0.1 every 5 epochs: epoch 1-5 = 0.1, epoch 6-10 = 0.01, etc.
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

model.train()
inputs = torch.randn(32, 10)
targets = torch.randint(0, 2, (32,))

for epoch in range(20):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Step the scheduler AFTER each epoch
    scheduler.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
```

### TensorBoard Integration

TensorBoard is a visualization toolkit for machine learning. It allows you to:
- **Track Metrics:** Plot loss and accuracy curves in real-time as training progresses.
- **Visualize Graphs:** Inspect the complete structure of your neural network.
- **Histogram Weights:** Monitor how weights and gradients shift across epochs to spot problems early.

In PyTorch, you use the `SummaryWriter` from `torch.utils.tensorboard`. After installing TensorBoard (`pip install tensorboard`), you run `tensorboard --logdir=runs` in your terminal to launch the dashboard.

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

# Creates a 'runs/experiment_1' log directory
writer = SummaryWriter('runs/experiment_1')

model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 2, (32,))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Log a scalar metric (loss) for this epoch
    writer.add_scalar('Loss/train', loss.item(), epoch)

# Log the model graph (requires a sample input)
writer.add_graph(model, inputs)
writer.close()
# Now run: tensorboard --logdir=runs
```

> **Key Term - `SummaryWriter`:** The PyTorch interface for writing TensorBoard log events. Each call to `add_scalar(tag, value, step)` records one data point on a named chart. Every training run should log to a uniquely named subdirectory so runs can be compared side by side.

### Training Callbacks

Callbacks are functions or objects that trigger at specific points in the training loop (e.g., `on_epoch_end`). They are used to inject logic like:
- **Early Stopping:** Stop training if the validation loss hasn't improved for X epochs.
- **Model Checkpointing:** Save the `state_dict` only when you reach a new "best" accuracy.
- **Logging:** Print status updates or send metrics to external services.

## Synthesizing the NLP Architecture

While the "Training Mastery" concepts (Schedulers, TensorBoard, Callbacks) apply to *any* neural network, the NLP architecture concepts introduced in the first half of this module are highly specific. 

A common point of confusion for beginners is how to connect an `Embedding` layer to an `LSTM`, and how to extract the final hidden state to pass to a regular `Linear` classifier. The code example below synthesizes these three NLP components into a single, cohesive architecture.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_classes):
        super().__init__()
        # 1. Embedding Layer: converts word integer IDs to dense vectors
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 2. LSTM Layer: processes the embedded sequence step by step
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)

        # 3. Classification head: maps the final hidden state to output classes
        self.fc = nn.Linear(hidden_dim, output_classes)

    def forward(self, x):
        # x shape: [Batch, SequenceLength]
        embedded = self.embedding(x)       # -> [Batch, SeqLen, EmbeddingDim]
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # h_n is the hidden state from the last time step: [1, Batch, HiddenDim]
        # squeeze removes the leading dimension
        final_hidden = h_n.squeeze(0)      # -> [Batch, HiddenDim]

        return self.fc(final_hidden)       # -> [Batch, OutputClasses]

# Assuming a vocabulary of 1000 words
model = SimpleLSTMModel(vocab_size=1000, embedding_dim=64, hidden_dim=128, output_classes=2)

# Set up optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# --- Early Stopping Callback Logic ---
best_loss = float('inf')
patience = 10
epochs_without_improvement = 0

print("Starting training...")
for epoch in range(50):
    # Dummy batch of 32 sentences (length 50) and 32 binary targets
    dummy_inputs = torch.randint(0, 1000, (32, 50))
    dummy_targets = torch.randint(0, 2, (32,))
    
    model.train()
    optimizer.zero_grad()
    predictions = model(dummy_inputs)
    loss = criterion(predictions, dummy_targets)
    loss.backward()
    optimizer.step()
    
    # Validation step / Early Stopping Check
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        epochs_without_improvement = 0
        # In a real script: torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1
        
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered at Epoch {epoch+1}! No improvement for {patience} iterations.")
        break
```

## Additional Resources

- [Understanding LSTMs (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch nn.LSTM Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch TensorBoard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
