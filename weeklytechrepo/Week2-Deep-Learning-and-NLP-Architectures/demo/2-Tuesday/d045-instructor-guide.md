# Demo: RNN Sequence Modeling and TensorBoard

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Recurrent Neural Network (RNN)** | *"If you want a model to understand the sentence 'The cat sat on the mat', does the order of words matter? What's the problem with feeding all words at once like a standard MLP?"* |
| **LSTM** | *"A simple RNN has memory like a goldfish — it often forgets things from several steps ago. If your model needs to link the pronoun 'it' to a noun from 10 words back, what architectural property would fix this?"* |
| **Embedding Layer** | *"The word 'king' is not a number, but neural networks only process numbers. How could you turn a word into a meaningful vector where 'king' and 'queen' are geometrically close to each other?"* |
| **Padding** | *"If one sentence has 3 words and another has 12 words, how do you group them into a single rectangular tensor for batch processing without corrupting the shorter sentence?"* |
| **TensorBoard** | *"If you're training a model overnight and want to know the next morning whether the loss was steadily decreasing or spiking erratically, what tool would you want? What's the alternative to reading thousands of raw log numbers?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/lstm-sequence-model.mermaid`.
2. Trace a sentence from raw text → token IDs → Embedding vectors → LSTM cells → final hidden state `h_n[0]` → Linear classifier → output logits.
3. Draw the **padding problem** side-by-side: a 3-word sentence vs a 6-word sentence as rectangular tensors, showing where `0` (PAD) tokens fill the shorter row.
4. **Discussion:** Ask the class: *"The LSTM returns `(lstm_out, (h_n, c_n))`. If you only care about classifying the whole sentence, which output do you use and why?"* (Answer: `h_n[0]` — the final hidden state is the LSTM's single compressed summary of the entire sequence.)

## Phase 2: The Code (Live Implementation)
**Time:** 25 mins
1. Open `code/d045-tensorboard-and-rnns.py`.
2. Walk through the `TextClassifierLSTM` class:
   - *Note: Highlight `nn.Embedding`. Ask: "Why do we set `padding_idx=0`?"* (Answer: it tells the network that PAD tokens carry no information — no gradient flows through them.)
   - Trace the `forward()` method shape-by-shape: `[B, S]` → `[B, S, E]` (Embedding) → LSTM → `hn[0]` `[B, H]` → Linear → `[B, C]`.
3. Walk through `demonstrate_lstm_pipeline()`:
   - Show the three variable-length sequences. Ask students: *"What shape will the padded batch be?"* (Predicted: `[3, 3]` — 3 sentences, max length 3.)
   - Execute and confirm the printed shapes match predictions.
4. Walk through `demonstrate_tensorboard_logging()`:
   - Highlight the `SummaryWriter` creation. Ask: *"Why would you use a unique log dir name like `runs/lstm_demo_v1` for each experiment?"* (Answer: so runs don't overwrite each other and can be compared side-by-side in the TensorBoard UI.)
   - Show the `add_scalar('Loss/train', loss, epoch)` call inside the loop.
   - After execution, run `tensorboard --logdir=runs` in the terminal and project the loss curve on screen.
   - *Note: This is the payoff — students see their first real-time loss curve. This is how professional ML engineers monitor production training jobs.*

## Summary
Reiterate that LSTMs solve the sequence-ordering problem that MLPs cannot, and that TensorBoard transforms raw loss numbers into actionable visual signals — an essential tool in any production ML workflow.
