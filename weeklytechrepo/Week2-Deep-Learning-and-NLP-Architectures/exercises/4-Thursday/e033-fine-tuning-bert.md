# Lab: Fine-Tuning BERT with Hugging Face

## The Scenario
Your company has decided to transition away from writing custom PyTorch LSTMs from scratch. Instead, you've been tasked with implementing a state-of-the-art Transformer model utilizing the Hugging Face `transformers` library. Your objective is to download a highly-optimized, pre-trained BERT variation (`distilbert-base-uncased-finetuned-sst-2-english`), tokenize a batch of incoming product reviews, and perform inference to classify their sentiment.

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e033-huggingface_lab.py`.
3. Complete the `initialize_nlp_pipeline` function:
   - Use `AutoTokenizer.from_pretrained` to download the tokenizer for `distilbert-base-uncased-finetuned-sst-2-english`.
   - Use `AutoModelForSequenceClassification.from_pretrained` to download the actual model weights for the same architecture.
   - Return both the `tokenizer` and the `model`.
4. Complete the `analyze_sentiment` function:
   - Call the tokenizer on the `reviews` list. You must pass `padding=True`, `truncation=True`, and `return_tensors="pt"` so that the outputs are ready for PyTorch.
   - Set the model to evaluation mode (`.eval()`).
   - Execute a forward pass using `torch.no_grad()`. Pass the inputs using dictionary unpacking: `**inputs`.
   - Apply a Softmax function to the `outputs.logits` along `dim=-1` to convert the raw logits into percentages (probabilities).
   - Use `torch.argmax` on probabilities to find the `predicted_classes` index (0 for Negative, 1 for Positive).

## Definition of Done
- The script executes successfully (the model loads from cache or downloads from Hugging Face Hub without error).
- The output clearly prints each of the three reviews alongside their predicted sentiment label (`POSITIVE` or `NEGATIVE`) and a confidence score derived from the Softmax probabilities.
- The `Raw Logits Shape` printed to console is `torch.Size([3, 2])` — confirming all 3 reviews were processed in a single batched forward pass.
