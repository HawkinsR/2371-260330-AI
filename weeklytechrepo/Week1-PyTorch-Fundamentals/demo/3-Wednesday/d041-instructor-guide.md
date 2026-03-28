# Demo: Custom Datasets and Dataloaders

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Mini-Batch** | *"If you were grading 1,000 student essays, would it be smarter to read all of them before writing any feedback, or to grade a stack of 30 and adjust your rubric as you go? Why?"* |
| **Sigmoid Function** | *"What if an AI needed to give you a YES or NO answer, but it wasn't 100% sure? How might it express 'I'm 80% confident this is a cat'?"* |
| **Softmax Function** | *"If a model must pick between 10 different animal categories, how is that different from a simple yes/no cat problem? What changes about the output?"* |
| **Class Imbalance** | *"Imagine a smoke detector trained on data that is 99.9% 'no fire' examples. Why would that detector be dangerously bad even if it's 99.9% accurate?"* |

## Phase 1: The Concept (Whiteboard/Diagram)

**Time:** 10 mins

1. Open `diagrams/dataset-dataloader-flow.mermaid`.
2. Address the Memory Problem:
   - "If we have a self-driving car dataset with 5 million images, we cannot load all 5 million into the `X` tensor like we did yesterday with 5 numbers. Our RAM will instantly crash."
   - Explain the `Dataset` class boundary. It only *knows* where the images are. It is the "librarian." When asked via `__getitem__`, it fetches **exactly one** book, applies data augmentation (Transforms), and returns a single tensor.
   - Explain the `DataLoader` class boundary. It acts as the "manager." It requests 32 individual books from the librarian, stacks them together into a massive block (a "batch"), and hands that batch securely to the model.

## Phase 2: The Code (Live Implementation)

**Time:** 30 mins

1. Open `code/d041-custom-dataset-and-transforms.py`.
2. **The Dataset Class (Lines 8-28):**
   - Point out that we inherit from `torch.utils.data.Dataset`.
   - Walk through `__len__` and `__getitem__`. Emphasize that these two methods are *mandatory constraints*. If they don't exist, PyTorch dataloaders will literally refuse to compile.
   - Briefly simulate reading a fake image using PIL inside `__getitem__`.
3. **The Transformers Pipeline (Lines 32-38):**
   - Dissect `transforms.Compose`.
   - Ask the class: "Why do we resize everything? (Because neural networks require rigid input shapes). Why do we RandomHorizontalFlip? (To artificially create more images and prevent overfitting)."
   - Emphasize `ToTensor()` — it magically converts a PIL image into a multi-dimensional array and scales pixel values from (0-255) to (0.0-1.0).
4. **The DataLoader Engine (Lines 47-53):**
   - Explain `batch_size=4` and `shuffle=True`. Discuss why `num_workers=0` (we are on Windows/basic CPU, but in cloud Linux boxes, we set this to 4 or 8 to dramatically speed up disk I/O).
5. **Execution (Lines 56-62):**
   - Run the script and observe the console output.
   - Track the shapes specifically: Point out that single items are `[3, 32, 32]` but loops print out `[4, 3, 32, 32]`. The 4 represents the batch index `N` which PyTorch injects at index 0. This is what the `nn.Module.forward()` loop expects!
