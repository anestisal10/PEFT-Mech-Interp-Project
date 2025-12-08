# Mechanistic Interpretability of Fine-Tuned GPT-2 for Summarization

This project investigates the internal mechanisms of language models ( like GPT-2, Bart etc) fine-tuned for summarization on (primarily) the CNN/DailyMail dataset. By employing **Edge Attribution Patching (EAP-IG)**, we extract and analyze "circuits" subgraphs of the model responsible for the task and track their evolution throughout the fine-tuning process.

> [!NOTE]
> For a detailed breakdown of (so far) numerical results, including ROUGE scores, circuit faithfulness, and layer-wise analysis, please refer to **[results.md](results.md)**.

## 🎯 Project Goals

The primary objective of this research is to bridge the gap between mechanistic interpretability and efficient fine-tuning. Specifically, we aim to:

1.  **Map the "Summarization Circuit"**: Identify which specific components (heads, MLPs) and connections (edges) in the models  become active or suppressed when learning to summarize.
2.  **Track Circuit Evolution**: Observe how this circuit changes from the base model to the fully fine-tuned state. Does the model learn by adding new nodes or by rewiring existing connections?
3.  **Optimize PEFT**: Use the insights from the circuit analysis to design "Targeted LoRA" strategies. Instead of applying LoRA to all layers, can we target only the "critical layers" identified by our analysis?
4.  **Comparisons**: Possibly compare findings between different models and or datasets.

## 🧪 Hypothesis & Assumptions

Our work is grounded in the following key hypothesis:

*   **The Edge-Centric Hypothesis**: We hypothesize that fine-tuning primarily alters the *connectivity* (edges) between model components rather than the components (nodes) themselves.
*   **Critical Layer Existence**: We assume that specific layers play a disproportionate role in the summarization task, and that these layers can be identified through differential analysis of hidden states and attention patterns.
*   **Transferability**: We assume that findings from GPT-2 Small can offer insights applicable to larger models (like BART) or different datasets (Multi-News).

## ⚙️ Methodology

Our experimental pipeline consists of five distinct stages:

### 1. Fine-Tuning with Checkpointing
We fine-tune a standard `gpt2-small` model on a subset of the CNN/DailyMail dataset. Crucially, we save checkpoints at frequent intervals to capture the gradual shift in the model's internal state. For already existing (and public) finetuned versions of models, this step is not necessary. 

### 2. Causal Data Generation
To isolate the "summarization" capability, we generate a specific dataset for interpretability. We create pairs of:
*   **Clean Input**: The original article.
*   **Corrupted Input**: The article with "salient" sentences (those with high ROUGE overlap with the summary) masked out.
This allows us to trace which parts of the model are responsible for recovering the summary information.

### 3. Differential Analysis
We compare the circuits and internal states (hidden states, attention patterns) between the base and finetuned models. We measure:
*   **Hidden State Similarity**: Uses cosine similarity to quantify the divergence of latent representations at each layer.
*   **Attention Entropy**: Measures information concentration, where decreased entropy indicates a shift toward focused information selection.
*   **KL Divergence**: Quantifies how significantly the attention distributions shift between the pre-trained and fine-tuned models.

### 4. Circuit Extraction (EAP-IG)
We use **Edge Attribution Patching with Integrated Gradients (EAP-IG)** to attribute the model's performance to specific edges in the computational graph. This yields a sparse "circuit" for each checkpoint.We also test the discovered circuits for their faithfulness (performing well enough in the task) and robustness (being immune to perturbations of the data).  


### 5. PEFT Bake-Off
Finally, we validate our findings by training new models using Low-Rank Adaptation (LoRA). We compare:
*   **Standard LoRA**: Applied to all layers.
*   **Targeted LoRA**: Applied *only* to the critical layers identified in step 3.
*   **Circuit LoRA**: Applied *only* to the critical layers identified in step 4.
*   **Random LoRA**: Applied to random layers (control).

---
*Author: Anestis*
