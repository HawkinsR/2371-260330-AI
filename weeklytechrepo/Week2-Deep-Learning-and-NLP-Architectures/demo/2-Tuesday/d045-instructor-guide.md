# Demo: Transfer Learning and Fine-Tuning

## Discussion Topics

Use these prompts to check student understanding before diving into the demo. Give students 30–60 seconds to think before calling on someone.

| Term | Prompt |
|---|---|
| **Transfer Learning** | *"If an expert chef spent 10 years learning French cuisine, how much easier would it be for them to learn Italian cuisine vs. someone who never cooked? What's the AI equivalent?"* |
| **Fine-Tuning** | *"If we hire that chef to cook Italian, do we want them to forget everything they know about chopping and sautéing? Which skills carry over and which need adjusting?"* |
| **Intersection over Union (IoU)** | *"If two overlapping circles represent a predicted mask and the true mask, how would you describe the quality of the prediction numerically using area?"* |
| **Dice Coefficient** | *"In medical imaging, a tumor might occupy 2% of a scan. Why would standard accuracy be useless here, and what property should our metric focus on?"* |

## Phase 1: The Concept (Whiteboard/Diagram)
**Time:** 10 mins
1. Open `diagrams/transfer-learning-flow.mermaid`.
2. Trace the path from the Pre-trained ResNet18 down to the Old Classifier. Explain that this model already knows how to detect robust features.
3. Transition to the **Fine-Tuning** subgraph. Point out the discarded old classifier and the newly attached classifier.
4. **Discussion:** Ask the class: "Why do we set `Requires Grad = False` on the frozen feature extractor?" (Answer: To save compute and prevent our chaotic initial gradients from destroying the pre-learned weights).

## Phase 2: The Code (Live Implementation)
**Time:** 20 mins
1. Open `code/d045-transfer-learning-resnet.py`.
2. Walk through `demonstrate_transfer_learning()`.
   - *Note: Highlight the `for param in model.parameters(): param.requires_grad = False` loop. This is the core of Transfer Learning.*
3. Execute the script. 
4. Point out the output where it explicitly lists the `params_to_update`. Verify that only `fc.weight` and `fc.bias` are listed.
5. Discuss the **Checkpointing** section, showing how `state_dict` is mapped, saved, and reloaded. Emphasize calling `.eval()` after loading for inference.

## Summary
Reiterate that Transfer Learning transforms computationally prohibitive tasks into feasible projects by standing on the shoulders of giants.
