# Demo: Transfer Learning and Fine-Tuning

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Transfer Learning** | *"If an expert chef spent 10 years mastering French cuisine, how much easier would it be for them to learn Italian compared to someone who has never cooked? What's the AI equivalent of all that accumulated cooking skill?"* |
| **Fine-Tuning** | *"If you hire that chef to cook Italian, do you want them to forget everything they know about knife skills and heat control? Which skills carry over, and which need adjusting for the new cuisine?"* |
| **Freezing Layers / requires_grad** | *"If we're only updating the final classification layer, why freeze the rest? What's the risk of letting our randomly-initialized new head's chaotic early gradients flow all the way back through the pre-trained backbone?"* |
| **Intersection over Union (IoU)** | *"If I give you two overlapping circles — one representing a predicted mask and one the ground-truth mask — how would you score the quality of that prediction using only area math?"* |
| **state_dict / Checkpointing** | *"If training crashes at epoch 47 out of 100, what would you need to have been saving regularly to avoid starting over from scratch?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/transfer-learning-flow.mermaid`.
2. Trace the pre-trained ResNet18 down to its original 1000-class head. Explain this model already knows how to detect edges, textures, and shapes — trained on ImageNet's 1.2 million images.
3. Transition to the **Fine-Tuning** subgraph. Point out: the backbone is frozen (locked), the new head is the only trainable component.
4. **Discussion:** Ask the class: *"Why use `weights=ResNet18_Weights.DEFAULT` instead of `pretrained=True`?"* (Answer: `DEFAULT` is the modern, explicit API — `pretrained=True` is deprecated and may silently resolve to unexpected weights in future PyTorch versions.)

## Phase 2: The Code (Live Implementation)
**Time:** 25 mins
1. Open `code/d046-transfer-learning-masterclass.py`.
2. Walk through `demonstrate_transfer_learning()` step by step:
   - **Loading:** Print `model.fc` before modification. Show students the default `Linear(512, 1000)` head — 1000 ImageNet classes.
   - **Freezing:** Highlight the `param.requires_grad = False` loop. Ask: *"If we freeze all params and then attach a new `fc` layer, why does the new layer still train?"* (Answer: Newly created tensors default to `requires_grad=True`.)
   - **Head replacement:** Show `model.fc = nn.Linear(num_ftrs, 5)`. Print the new head to confirm the output shrunk from 1000 → 5.
3. Execute the script. Point out the parameter list — **only `fc.weight` and `fc.bias` are trainable**. Ask: *"If a full ResNet18 has 11 million parameters, how many are we actually updating?"* (Answer: just 512×5 + 5 = 2,565.)
4. Discuss the **Checkpointing** section:
   - Explain `state_dict()` is an `OrderedDict` of tensor values — not architecture, not optimizer state. It's the minimum needed to reproduce a model's predictions.
   - Walk the **reload pattern**: create empty shell → adapt head → `load_state_dict()` → `.eval()`.
   - *Note: Emphasize `.eval()` after loading. Forgetting it is one of the most common silent bugs in production — BatchNorm and Dropout behave differently in training mode.*

## Summary
Reiterate that Transfer Learning turns computationally prohibitive tasks into feasible projects by standing on the shoulders of giants. The `state_dict` save/load pattern is the foundation of every production ML deployment workflow — from SageMaker checkpointing to model versioning in the registry.
