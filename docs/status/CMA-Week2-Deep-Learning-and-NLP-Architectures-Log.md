# Weekly Epic: Transition into advanced deep learning frameworks focusing on Computer Vision with CNNs and NLP with RNNs and Transformers

## 1-Monday

### Written Content

- [ ] Create `c246-cnn-and-module-architecture.md`: CNN architecture intuition, Convolutions (`nn.Conv2d`) & Filters, Pooling Layers (`MaxPool`) & Stride, Feature Maps Visualization, Modern Architectures (ResNet Intuition), `nn.Module` Class Structure & `forward()`, MLPs & Intuition.

### Instructor Demo

- [ ] Create `d044-building-a-cnn.py`: Build a custom ResNet-style block or a standard MLP vs CNN architecture comparison using the `nn.Module` class.

### Trainee Exercise

- [ ] Create `e030-cnn-image-classification.md`: Construct a functional CNN with convolution and pooling layers for basic image classification tasks.

## 2-Tuesday

### Written Content

- [ ] Create `c247-training-mastery-and-rnns.md`: RNNs for Sequence Data, Schedulers & LR Decay, TensorBoard Setup & Integration, Logging Metrics & Visualizing Graphs, Training Callbacks.

### Instructor Demo

- [ ] Create `d045-tensorboard-and-rnns.py`: Demonstrate training an initial RNN and visualizing the loss curves and metric scalars in real-time via TensorBoard.

### Trainee Exercise

- [ ] Create `e031-implementing-rnn-mastery.md`: Configure learning rate schedulers and integrate TensorBoard logging into a basic sequence prediction training run.

## 3-Wednesday

### Written Content

- [ ] Create `c248-advanced-cv-and-transfer-learning.md`: LSTMs & GRUs, nn.Embedding Layers, Handling Variable Length Sequences, Object Detection and Semantic Segmentation, Segmentation metrics (Dice Coefficient, IoU), Loading Pre-trained Models, Freezing Layers & Fine-tuning, `state_dict` & Checkpointing.

### Instructor Demo

- [ ] Create `d046-transfer-learning-masterclass.py`: Fine-tune a pre-trained CV model (e.g., ResNet) for segmentation and an LSTM model for sequences using frozen layers.

### Trainee Exercise

- [ ] Create `e032-advanced-cv-fine-tuning.md`: Practice freezing layers, fine-tuning pre-trained image models for object detection/segmentation, and managing checkpoints.

## 4-Thursday

### Written Content

- [ ] Create `c249-transformers-and-attention.md`: Transformer Architecture Details (Encoder/Decoder), Attention Mechanisms, BERT, GPT & T5 Use Cases, Hugging Face `transformers` Library, `AutoTokenizer` & `AutoModel`, Fine-tuning.

### Instructor Demo

- [ ] Create `d047-huggingface-bert-inference.py`: Tokenize text, perform inference with pre-trained BERT, and extract context embeddings using Hugging Face.

### Trainee Exercise

- [ ] Create `e033-fine-tuning-bert.md`: Download a pretrained Hugging Face auto-model, configure the tokenizer, and generate prediction outputs.

## 5-Friday

### Written Content

- [ ] Create `c250-advanced-mechanics-and-safety.md`: Training Mechanics (AMP, Gradient Clipping), Reproducibility patterns, PII Masking with Presidio, LLM Lingua (Context Compression), Explainability with SHAP values, Algorithm selection framework.

### Instructor Demo

- [ ] Create `d048-mixed-precision-and-callbacks.py`: Demonstrate training optimization with CUDA Mixed Precision Training (AMP) and Gradient Clipping.

### Trainee Exercise

- [ ] Create `e034-implementing-early-stopping.md`: Wrap a basic training loop with early-stopping heuristics, callback architectures, and basic SHAP visualizations.
