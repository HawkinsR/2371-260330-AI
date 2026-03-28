# Demo: Transformers and Hugging Face

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Large Language Model (LLM)** | *"ChatGPT, Claude, Gemini — what do you think they all have in common architecturally? What does 'Large' actually mean in Large Language Model?"* |
| **Tokenization** | *"Before a model can process the word 'unhappiness', it needs to convert it to numbers. But how does a model handle a word it's never seen before? What's a creative solution?"* |
| **Self-Attention** | *"In the sentence 'The trophy didn't fit in the bag because it was too big', what does 'it' refer to? How would a model need to look at the whole sentence to figure that out?"* |
| **Encoder vs. Decoder** | *"If BERT is great at understanding text and GPT is great at generating text, why would a translation system need both? What does each half contribute?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/transformer-architecture.mermaid`.
2. Flow overview: Start from the Input text and trace down to the `AutoTokenizer`. Explain that unlike standard vocabularies, Tokenizers often use "subwords" (e.g., splitting "revolutionized" into "revolution" + "ized").
3. Transition to the **Self-Attention Mechanism** subgraph. Describe the concept of Query, Key, and Value vectors as a database retrieval metaphor.
4. **Discussion:** Ask the class: "If LSTMs struggle with long paragraphs because they process linearly, how do Transformers fix this?" (Answer: By processing all words in parallel and computing Attention Scores universally).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d047-huggingface-bert-inference.py`.
2. Walk through `demonstrate_huggingface_pipeline()`.
   - Take a moment on the `inputs = tokenizer(...)` line. Highlight how much heavy lifting Hugging Face is doing by automatically formatting the `input_ids` and `attention_mask`.
3. Execute the script. 
4. Emphasize the `Tokens Breakdown` print loop. Show the class how the special tokens (`[CLS]` and `[SEP]`) are injected automatically.
5. In the **Bonus** embedding extraction phase, highlight the final tensor shape `[batch_size, sequence_length, hidden_dimension]`. This is the Context Matrix represented in the diagram.

## Summary
Reiterate that Hugging Face empowers developers by abstracting the enormous mathematical complexity of Self-Attention behind a clean, unified Python API.
