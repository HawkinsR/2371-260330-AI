# Demo: LSTM Sequence Prediction

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **NLP (Natural Language Processing)** | *"What's the hardest part of teaching a computer to understand a sentence versus teaching it to recognize a cat in a photo? What makes language uniquely complex?"* |
| **Hidden State** | *"Imagine reading a novel one word at a time and writing a one-sentence summary after each word. That summary is like the hidden state. How would the summary change with each new word?"* |
| **Word Embedding** | *"Why can't we just assign 'cat'=1, 'dog'=2, 'banana'=3? What math problem would arise if a model tried to use those numbers in arithmetic?"* |
| **LSTM Gates (Forget, Input, Output)** | *"When you're reading a long document, your brain forgets unimportant details and remembers crucial facts. Which LSTM gate handles each of those tasks?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/lstm-sequence-flow.mermaid`.
2. Trace the path from Raw Text to the `Embedding Layer`. Emphasize that raw integers are converted to meaningful dense vectors.
3. Walk through the **LSTM Time Steps** subgraph. Explain the sequential nature of reading Word 1, generating a hidden state, and passing that state to the step for Word 2.
4. **Discussion:** Ask the class: "Why does an LSTM need an explicitly defined `padding_idx` in its Embedding layer?" (Answer: To logically ignore the synthetic zeros we add to match sequence lengths within the batch).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d046-lstm-sequence-prediction.py`.
2. Walk through the `demonstrate_lstm_pipeline()` function.
   - Point out how `pad_sequence` handles the variable lengths, forcefully aligning the tensors. Take time to show the printed `Padded Batch Tensor`.
3. Review the `TextClassifierLSTM` class architecture.
   - *Note: Highlight that the output we care about for classification is `hn[0]` (the distinct hidden state preserved from the final token passed through the recurrent unit).*
4. Execute the script, step by step, demonstrating the dimensional transformations (`[batch, seq_len] -> [batch, seq_len, embed_dim] -> [batch, num_classes]`).

## Summary
Reiterate that LSTMs conquer the "amnesia" of standard RNNs while maintaining critical sequential memory spanning long bodies of text.
