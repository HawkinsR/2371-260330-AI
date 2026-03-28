# Sequence Processing and RNNs

## Learning Objectives

- Understand why standard feedforward networks cannot process sequential data like text.
- Map discrete words to continuous numeric vectors using `nn.Embedding` layers.
- Explain the internal feedback loop that defines Recurrent Neural Networks (RNNs).
- Describe how LSTMs and GRUs solve the vanishing gradient problem in long sequences.
- Build a data pipeline that handles variable-length sequences.

## Why This Matters

Images have fixed dimensions (e.g., 224×224 pixels), but text does not. A sentence can be 5 words or 500. A standard feedforward network processes its entire input at once with no sense of order—the word "not" appearing before "good" versus after it produces the same output, which is clearly wrong for understanding meaning.

Recurrent Neural Networks process sequences one step at a time, maintaining an internal hidden state that accumulates context as each word is read. This makes RNNs the foundational architecture for Natural Language Processing (NLP), enabling tasks like sentiment analysis, text classification, and machine translation.

## The Concept

### Word Embeddings

Machine learning models work with numbers, not strings. The simplest approach is to assign a unique integer to each word (e.g., "apple" = 1, "banana" = 2). The problem is that integers imply a mathematical relationship that does not exist—"banana" is not twice "apple."

An `nn.Embedding` layer solves this by mapping each word integer to a dense floating-point vector of a fixed size (e.g., 64 numbers). These vectors are learned during training. Words that appear in similar contexts end up with similar vectors. The word "king" and "queen" might be geometrically close in the embedding space, while "king" and "keyboard" are far apart.

The key advantage is that the model learns *meaningful relationships between words* entirely from the data, without any human-defined rules.

> **Key Term - Word Embedding:** A learned, dense numeric vector that represents a single word. Each dimension of the vector captures some aspect of the word's meaning. Embeddings are learned during training so that semantically similar words end up near each other in the vector space.

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

## Code Example

```python
import torch
import torch.nn as nn

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

# Batch of 32 sentences, each 50 words long (represented as integer token IDs)
dummy_text_batch = torch.randint(0, 1000, (32, 50))
predictions = model(dummy_text_batch)
print("Output shape:", predictions.shape)  # [32, 2]
```

## Additional Resources

- [Understanding LSTMs (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch nn.LSTM Docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
