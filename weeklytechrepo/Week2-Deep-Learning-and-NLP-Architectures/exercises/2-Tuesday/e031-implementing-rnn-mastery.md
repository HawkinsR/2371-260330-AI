# Lab: Text Classification with LSTMs

## The Scenario
You are developing an automated moderation filter for a chat application. The filter needs to read incoming messages and classify them as "Safe" (0) or "Toxic" (1). Since chat messages can be anything from a single word to an entire paragraph, you must build a model that can handle variable-length sequences using an Embedding layer and an LSTM backbone.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e031-rnn-mastery.py`.
3. Complete the `ModerationLSTM` class:
   - Inherit from `nn.Module`.
   - Initialize an `nn.Embedding` layer. Look at the constructor arguments to pass in the `vocab_size` and `embedding_dim`. Set `padding_idx=0`.
   - Initialize an `nn.LSTM` layer. Set `input_size` to `embedding_dim`, `hidden_size` to `hidden_dim`, and crucially, set `batch_first=True`.
   - Initialize an `nn.Linear` fully connected layer to map the `hidden_dim` to the `output_classes`.
   - In the `forward` pass, pass the input `x` through the embedding layer.
   - Pass the embedded sequence through the LSTM.
   - Extract the hidden state from the final time step. *Hint: The LSTM returns `lstm_out, (h_n, c_n)`. `h_n` has shape `[num_layers, batch_size, hidden_dim]`. For a single-layer LSTM, `h_n[0]` selects the first (only) layer's hidden state, giving a tensor of shape `[batch_size, hidden_dim]` — exactly what the linear head expects.*
   - Pass that final hidden state through the linear classification head and return the result.
4. Complete the `process_chat_logs` function:
   - Use PyTorch's `pad_sequence` to pad the list of variable-length Tensors. Make sure `batch_first=True` and `padding_value=0`.
   - Pass the padded batch through the model to get the predictions.
5. Add TensorBoard Logging:
   - Import `SummaryWriter` from `torch.utils.tensorboard`.
   - Create a `SummaryWriter` instance (e.g., `writer = SummaryWriter('runs/moderation_lstm')`).
   - Simulate 5 training epochs with a decaying loss (e.g., `loss = 1.0 / (epoch + 1)`) and log each epoch's loss using `writer.add_scalar('Loss/train', loss, epoch)`.
   - Call `writer.close()` when done and print the directory where logs were saved.

## Definition of Done
- The script executes without crashing.
- The shapes printed at the end of the script exactly match the expected sizes:
  - Padded Batch Shape: `torch.Size([4, 6])` (4 sequences, max length 6)
  - Output Predictions Shape: `torch.Size([4, 2])` (4 sequences, 2 classes)
- The console prints `"TensorBoard logs saved to: runs/moderation_lstm"` (or your chosen log dir) and a `runs/` directory appears in your working folder.
