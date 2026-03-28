# Weekly Epic: Transition into advanced deep learning frameworks focusing on Computer Vision with CNNs and NLP with RNNs and Transformers

## 1-Monday

### Written Content

- [x] Create `c246-cnn-architecture-and-feature-maps.md`: CNN architecture intuition, Convolutions (`nn.Conv2d`) & Filters, Pooling Layers (`MaxPool`) & Stride, Feature Maps Visualization, Modern Architectures (ResNet Intuition).

### Instructor Demo

- [x] Create `d044-building-a-cnn.py`: Build a custom ResNet-style block and visualize the initial feature maps of a given convolution.

### Trainee Exercise

- [x] Create `e030-cnn-image-classification.md`: Construct a functional CNN with convolution and pooling layers for basic image classification tasks.

## 2-Tuesday

### Written Content

- [x] Create `c247-advanced-vision-and-transfer-learning.md`: Advanced Vision & Transfer Learning, Object Detection and Semantic Segmentation, Segmentation metrics: Dice Coefficient, IoU, Loading Pre-trained Models, Freezing Layers & Fine-tuning, `state_dict` & Saving/Loading Models, Checkpointing & Resuming Training.

### Instructor Demo

- [x] Create `d045-transfer-learning-resnet.py`: Load a pre-trained ResNet model, freeze the feature extractor, and fine-tune the classification head on a custom dataset.

### Trainee Exercise

- [x] Create `e031-fine-tuning-pretrained-models.md`: Practice freezing layers, fine-tuning pre-trained image models, and computing IoU/Dice coefficient on simple segmentation maps.

## 3-Wednesday

### Written Content

- [x] Create `c248-sequence-processing-and-rnns.md`: RNN and LSTM sequences, Sequence Processing Basics, Word embeddings and vector representations, `nn.Embedding` Layers, Recurrent Neural Networks (RNN), LSTM/GRU Cells & Gating, Handling Variable Length Sequences.

### Instructor Demo

- [x] Create `d046-lstm-sequence-prediction.py`: Pad variable length sequences and pass them iteratively through an embedding and LSTM cell.

### Trainee Exercise

- [x] Create `e032-text-classification-with-lstm.md`: Construct a simple NLP vocabulary, generate embeddings, and classify text sequences using GRUs/LSTMs.

## 4-Thursday

### Written Content

- [x] Create `c249-transformers-and-huggingface.md`: Transformer and Attention mechanisms, Transformer Intuition (Encoder/Decoder), Hugging Face `transformers` Library, `AutoTokenizer` & `AutoModel`, Using Pre-trained BERT/GPT.

### Instructor Demo

- [x] Create `d047-huggingface-bert-inference.py`: Tokenize text, perform inference with pre-trained BERT, and extract context embeddings using Hugging Face.

### Trainee Exercise

- [x] Create `e033-fine-tuning-bert.md`: Download a pretrained Hugging Face auto-model, configure the tokenizer, and generate prediction outputs.

## 5-Friday

### Written Content

- [x] Create `c250-advanced-training-mechanics.md`: Training Mechanics, Reproducibility patterns, Mixed Precision Training (AMP), Gradient Clipping, Early Stopping Logic, Training Callbacks, Explainability with SHAP values, Algorithm selection framework.

### Instructor Demo

- [x] Create `d048-mixed-precision-and-callbacks.py`: Demonstrate training optimization with CUDA Mixed Precision Training (AMP) and Gradient Clipping to boost efficiency.

### Trainee Exercise

- [x] Create `e034-implementing-early-stopping.md`: Wrap a basic training loop with early-stopping heuristics, callback architectures, and basic SHAP visualizations.
