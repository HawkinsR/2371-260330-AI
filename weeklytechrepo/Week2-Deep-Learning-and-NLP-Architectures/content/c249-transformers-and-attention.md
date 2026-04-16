# Transformers and Attention

## Learning Objectives

- Grasp the core mechanism of Transformers and Attention.
- Understand the Transformer Intuition balancing Encoder and Decoder blocks.
- Navigate the open-source Hugging Face `transformers` Library ecosystem.
- Implement the `AutoTokenizer` to structure raw text for model ingestion.
- Deploy Pre-trained BERT/GPT models via the `AutoModel` interface.
- Identify BERT, GPT, and T5 use cases for specific NLP tasks.

## Why This Matters

For decades, RNNs and LSTMs were the standard for sequential text logic. However, recurrent models must process text linearly (word by word), preventing parallel computation and limiting their ability to remember context from massive documents. The "Attention Is All You Need" paper introduced the Transformer structure, replacing recurrence entirely with "Self-Attention." This breakthrough allowed parallel processing on GPUs, birthing Large Language Models (LLMs) like BERT and GPT.

> **Key Term - Large Language Model (LLM):** A deep learning model with billions of parameters, trained on vast amounts of text, capable of understanding and generating human-like language. Examples include BERT, GPT-4, and Claude. LLMs became practical because the Transformer architecture allowed training to be parallelized across thousands of GPUs simultaneously.

## The Concept

### Transformer Intuition (Encoder/Decoder)

The original Transformer was designed for translation (e.g., English to French). It has two halves:

- **Encoder:** Reads the entire English input sentence simultaneously and generates a dense mathematical context matrix representing the meaning of the phrase using Self-Attention.
- **Decoder:** Takes that context matrix and generates the French words one by one, focusing its "Attention" dynamically on different parts of the English context depending on what French word it is currently writing.
Modern Architectures often split this. **BERT** is Encoder-only (excellent at reading and classifying text). **GPT** is Decoder-only (excellent at generating and predicting the next word).

> **Key Term - Tokenization:** The process of splitting raw text into smaller units called **tokens** before feeding it to a language model. A token is roughly equivalent to a word or word-fragment (e.g., "unhappiness" might become ["un", "##happiness"]). Every unique token has a corresponding integer ID that the model reads as input. When you call `tokenizer(text, return_tensors="pt")`, it returns a dictionary with two key tensors: `input_ids` (the integer token IDs) and `attention_mask` (a binary mask of 1s and 0s indicating which positions are real tokens vs. padding to be ignored).

### Attention Mechanisms

"Self-Attention" allows a word to look at all other words in a sentence and decide mathematically which ones are relevant to its current meaning. For example, in the sentence "The animal didn't cross the street because it was too tired," Attention allows the word "it" to strongly associate with "animal" rather than "street."

> **Self-Attention (Attention Mechanism):** A mechanism that allows each word (token) in a sequence to compute a weighted relationship with every other word in the same sequence simultaneously. Instead of reading word-by-word like an RNN, the entire sentence is processed in parallel, and each word dynamically "attends to" the most relevant other words. This is what allows Transformers to handle long-range dependencies in text without vanishing gradients.

### Positional Encoding

Because Transformers process every token in a sequence simultaneously (not sequentially like RNNs), they have no built-in sense of word order. The sentence "the dog bit the man" and "the man bit the dog" would produce identical attention outputs without intervention.

**Positional Encodings** solve this by injecting a unique numeric "position signal" into each token's embedding before it enters the Transformer. The original paper used fixed sine and cosine functions of different frequencies; modern models (like BERT and GPT) learn the position embeddings during training. Either way, the result is the same: the model can distinguish "bank" at position 2 from "bank" at position 15 and attend to each appropriately.

> **Key Term - Positional Encoding:** A vector added to each token embedding that encodes the token's position in the sequence. Without this, a Transformer treats the input as an unordered set of words. Positional encodings restore word-order awareness, allowing the model to distinguish meaning based on where words appear in a sentence.

### Model Use Cases (BERT, GPT, T5)

The Transformer architecture has been adapted into three main "flavors" for different industry tasks:

1.  **BERT (Bidirectional Encoder Representations from Transformers):**
    *   **Architecture:** Encoder-only.
    *   **Best For:** Natural Language Understanding (NLU). Tasks like **Sentiment Analysis**, **Named Entity Recognition (NER)**, and **Question Answering**. It looks at the context both to the left and right of a word.
2.  **GPT (Generative Pre-trained Transformer):**
    *   **Architecture:** Decoder-only.
    *   **Best For:** Natural Language Generation (NLG). Tasks like **Creative Writing**, **Code Generation**, and **Chatbots**. It focuses on predicting the next token in a sequence.
3.  **T5 (Text-To-Text Transfer Transformer):**
    *   **Architecture:** Full Encoder-Decoder.
    *   **Best For:** Task-to-task translation. It treats every problem as a text-to-text problem (e.g., "summarize: [text]" -> "[summary]"). Excellent for **Translation**, **Summarization**, and **Paraphrasing**.

### Hugging Face Ecosystem

Hugging Face democratized Transformers by providing an open-source hub of pre-trained weights. Instead of manually writing PyTorch `nn.Module` classes for complex attention heads, we can use their `AutoModel` and `AutoTokenizer` classes to instantly pull down state-of-the-art models from the cloud in three lines of code.

### Fine-Tuning a Pre-Trained Transformer

Downloading a pre-trained model for inference is powerful, but the real value comes from **fine-tuning**: taking a general model like BERT and adapting it to your specific task by training it for a few additional epochs on your own labeled dataset.

The workflow is:
1. **Load** a pre-trained model with `AutoModelForSequenceClassification` (or similar task-specific class).
2. **Freeze or unfreeze** layers — in practice, most fine-tuning runs train *all* the layers at a very low learning rate (e.g., `2e-5`), rather than freezing anything.
3. **Train** for a small number of epochs (typically 2–5). Because the model already knows language, it converges quickly.
4. **Evaluate and save** using `model.save_pretrained()` and `tokenizer.save_pretrained()`.

Fine-tuning is covered in depth in Thursday's exercise (`e033`). The key intuition to carry forward is: a pre-trained transformer already understands grammar, context, and common-sense language patterns. Your fine-tuning run simply teaches it to *apply* that understanding to your specific labels.

> **Key Term - Fine-Tuning:** Continuing the training of a pre-trained model on a new, smaller, task-specific dataset. Because the model's weights are already a strong starting point, the model adapts quickly — often achieving state-of-the-art results with only a few hundred labeled examples.

## Code Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the model name from the Hugging Face Hub
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 1. The Tokenizer: Converts string text into integer IDs that BERT understands
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. The AutoModel: Downloads the PyTorch weights and architecture
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3. Preparation: Tokenize input and convert to PyTorch Tensors
text = "Transformers completely revolutionized the AI industry!"
inputs = tokenizer(text, return_tensors="pt")

# 4. Inference
with torch.no_grad():
    outputs = model(**inputs)
    
# The output logits can be mapped (e.g., Positive vs Negative sentiment)
predicted_class = torch.argmax(outputs.logits, dim=-1)
print(f"Predicted Class ID: {predicted_class.item()}")
```

## Additional Resources

- [The Illustrated Transformer (Jay Alammar)](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers Course](https://huggingface.co/course/chapter1/1)
- [Hugging Face Fine-Tuning a Pre-Trained Model](https://huggingface.co/docs/transformers/training)
