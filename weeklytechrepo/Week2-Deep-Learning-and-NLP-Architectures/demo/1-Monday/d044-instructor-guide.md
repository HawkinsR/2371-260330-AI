# Demo: Building a CNN and ResNet Block

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Convolutional Filter / Kernel** | *"If you wanted to detect vertical edges in a photo using math, what kind of small pattern matrix would you slide across each row of pixels?"* |
| **Feature Map** | *"After passing an image through a filter that detects edges, what would the output look like — bright where edges are, dark elsewhere. What would you call that output?"* |
| **Stride** | *"If a filter checks every single pixel (stride=1) vs. skipping every other pixel (stride=2), what's the tradeoff between coverage and speed?"* |
| **Translational Invariance** | *"If a model trained on cats only ever saw cats in the center of images, would it recognize a cat in the corner? What does 'translational invariance' mean for solving this?"* |
| **Vanishing Gradient** | *"If you're passing a message down a 100-person chain and each person tells the next person only half of what they heard, what happens to the original message by the time it reaches person 100?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 15 mins
1. Open `diagrams/cnn-architecture.mermaid`.
2. Flow overview: Trace the path from the Input Image through the `Conv2d` layer. Discuss how the filter sliding across the input produces the feature map.
3. **Discussion:** Ask the class, "If an image is 224x224 and we apply MaxPool(2x2), what happens to our spatial dimensions?"
4. Transition to the **ResNet Intuition** subgraph. Show how the "Residual / Skip Connection" bypasses the main convolution layers, combating the Vanishing Gradient Problem.

## Phase 2: The Code (Live Implementation)
**Time:** 25 mins
1. Open `code/d044-building-a-cnn.py`.
2. First, walk through the `visualize_feature_maps` function. 
   - *Note: Emphasize the shape outputs (`[Batch, Channels, Height, Width]`). Reference the `Conv2d` and `MaxPool2d` operations from the theory.*
3. Second, live-code or walk through the `ResNetBlock` class.
   - *Note: Highlight the `self.shortcut` logic in the init parameters and how it is explicitly added back to the main path (`out += identity`) inside the forward method.*
4. Execute the script to show the shapes dynamically changing.

## Summary
Reiterate that the goal of a CNN is feature extraction through convolutions, while ResNet blocks enable scaling these layers deeply without losing signal.
