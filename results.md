# Experimental Results

This document aggregates the numerical findings from the mechanistic interpretability and fine-tuning experiments so far.

## 1. Fine-tuning Performance (GPT-2 Small)

Comparison of the base GPT-2 model versus the model fine-tuned on the CNN/DailyMail subset (~20k samples).

| Metric | Base Model | Finetuned (20k) | Delta |
| :--- | :--- | :--- | :--- |
| **ROUGE-1** | 18.15 | **24.80** | +6.65 |
| **ROUGE-2** | 3.00 | **5.65** | +2.65 |
| **ROUGE-L** | 12.99 | **16.20** | +3.21 |
| **ROUGE-Lsum**| 15.22 | **22.91** | +7.69 |

*Note: Results based on 500 test samples.*

For comparison the paper's github page reported these results (finetuning and testing on the full dataset)

| Metric | Base Model | Finetuned (20k) | Delta |
| :--- | :--- | :--- | :--- |
| **ROUGE-1** | 18.7 | **23.4** | +4.7 |
| **ROUGE-2** | 1.4 | **3.5** | +2.1 |
| **ROUGE-L** | 16.3 | **20.6** | +4.3 |

## 2. Circuit Analysis

We extracted circuits using Edge Attribution Patching (EAP-IG) and evaluated their faithfulness (how well they recover full model performance) and robustness.

### Faithfulness
| Model | Circuit KL Divergence | Faithfulness Score |
| :--- | :--- | :--- |
| **Base Model** | 0.0000 | **92.98%** |
| **Finetuned** | 0.0000 | **89.38%** |

### Robustness (Stability)
Jaccard similarity of edges between different runs/checkpoints.

| Comparison | Jaccard Similarity | Interpretation |
| :--- | :--- | :--- |
| **C_after vs C_after2** | 0.8079 | High stability in finetuned circuits |
| **C_before vs C_before2** | 0.7123 | Moderate stability in base circuits |
| **C_before vs C_after** | **0.2848** | **Low similarity indicates significant circuit rewiring during fine-tuning** |

## 3. Differential Analysis (Layer-wise)

We analyzed how internal representations changed after fine-tuning.

### Top Changed Layers
This is a somewhat confusing result because it is different from what the original paper reported, but it can be attributed to the lower data samples.
*   **Hidden State Divergence**: Layer **11** changed the most (Distance: 0.56), followed by Layer 10 and 9.
*   **Attention Divergence (KL)**: Layer **5** changed the most (0.37), followed by Layer 7 and 6.
*   **Entropy Increase**: Entropy increased (contrary to the paper) in **all 12 layers** after fine-tuning, suggesting the model became less "sharp" or more distributed in its attention.

## 4. PEFT Bake-off Results

Comparison of different Parameter-Efficient Fine-Tuning (LoRA) strategies.

| Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L | Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| **Standard LoRA** (All Layers) | 18.77 | 6.48 | 15.05 | 1.86% |
| **Random Control** | **18.84** | **6.50** | **15.15** | 1.86% |
| **Circuit LoRA** (Layers 3,4,6,8,11) | 13.88 | 4.55 | 11.32 | 3.06% |
| **Targeted LoRA** (Layers 2,3,5) | 16.04 | 5.37 | 12.56 | **0.47%** |

### Key Observations
*   **Standard & Random** performed best, likely due to higher capacity coverage.
*   **Targeted LoRA** (using layers 2, 3, 5) achieved respectable performance with **~4x fewer parameters** (0.47% vs 1.86%).
*   **Critical Layers Identified**:
    *   Via Hidden States: 9, 10, 11
    *   Via Attention: 5, 6, 7
    *   Via Circuit Edges: 3, 4, 6, 8, 11

*Recommendation: A hybrid strategy targeting layers 5, 10, and 11 may offer the best balance between depth and performance.*

## 5. Additional Experiments

We extended the analysis to other models and datasets to verify the universality of our findings.

### 5.1. BART-Large (CNN/DailyMail)
*   **Entropy Change**: Entropy decreased in **12/12 layers** after fine-tuning.
*   **Critical Layers**:
    *   **Hidden State Divergence**: Layers **11, 10, 9** changed the most.
    *   **Attention Divergence**: Layers **11, 10, 8** changed the most.

### 5.2. GPT-2 (Multi-News Dataset)
*   **Entropy Change**: Entropy decreased in **8/12 layers**.
*   **Critical Layers**:
    *   **Hidden State Divergence**: Layers **11, 10, 9** changed the most.
    *   **Attention Divergence**: Layers **11, 10, 9** changed the most.

### Summary of Critical Layers Across Experiments
| Experiment | Top Changed Layers (Hidden State) | Top Changed Layers (Attention) |
| :--- | :--- | :--- |
| **GPT-2 (CNN/DM)** | 11, 10, 9 | 5, 7, 6 |
| **BART-Large (CNN/DM)** | 11, 10, 9 | 11, 10, 8 |
| **GPT-2 (Multi-News)** | 11, 10, 9 | 11, 10, 9 |

*Consistent Finding: The last layers (9-11) consistently show the largest shift in hidden state representations across all models and datasets.*
