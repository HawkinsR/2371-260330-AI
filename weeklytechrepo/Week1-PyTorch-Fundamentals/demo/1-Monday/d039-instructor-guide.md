# Demo: Tensor Operations and Autograd

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Hardware Acceleration** | *"Has anyone used a dedicated graphics card for gaming? What makes a GPU different from a CPU for number-crunching tasks?"* |
| **Gradient** | *"If you were hiking in the mountains, how would you find the fastest route down to the valley? What information would you need?"* (Connect: the gradient is that information for a model.) |
| **Directed Acyclic Graph (DAG)** | *"Can anyone draw what an assembly line looks like as a flowchart? Each station transforms a product and passes it forward — that is a DAG."* |
| **Leaf Node** | *"In a family tree, who are the 'leaf' members? Why would a training algorithm care about who the original inputs are?"* |

## Phase 1: The Concept (Whiteboard/Diagram)

**Time:** 15 mins

1. Open `diagrams/autograd-graph.mermaid` (using a Markdown previewer or Mermaid live editor).
2. Explain the Computational Graph:
   - "Deep Learning is just a sequence of mathematical operations."
   - Trace the solid arrows: Input `x` is multiplied by weight `w`, then bias `b` is added. This is the **Forward Pass** resulting in our prediction `y`.
   - Trace the dotted arrows: Once we calculate the `Loss` (how wrong we were), PyTorch automatically flows backward down those arrows.
   - Explain that Autograd calculates how much `w` and `b` need to change to make the Loss smaller. That is the derivative (`dLoss/dw`).

## Phase 2: The Code (Live Implementation)

**Time:** 25 mins

1. Open `code/d039-tensor-operations-and-gpu.py`.
2. **PyTorch vs NumPy Bridge:**
   - Walk through lines 7-15.
   - Show how easy it is to convert existing Python data into PyTorch objects using `torch.from_numpy()`.
3. **Tensor Operations:**
   - Run the script and highlight the output of addition and the dot product. Tensors act mostly like standard matrices.
4. **GPU Mechanics:**
   - Emphasize the `device = torch.device(...)` pattern (lines 28-34).
   - Explain that tensors "live" in memory. If `tensor_a` is on CPU and `tensor_b` is on GPU, you cannot add them together. They must both `.to(device)`.
5. **Autograd Mechanics (Connecting back to the Diagram):**
   - Point to lines 38-40. "We are setting `requires_grad=True`. We are telling PyTorch: 'Watch these variables and build that Mermaid diagram in the background'."
   - Run the code. Show how calling `.backward()` instantly populates `.grad` attributes with the exact derivatives we conceptualized in Phase 1.
