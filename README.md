# Multimodal Hallucination Mitigation in Vision-Language Models via DPO
Official Implementation for my Final Year Project.

## Overview
Hallucination remains a critical challenge in vision-language models (VLMs), where generated responses contain content inconsistent with the provided image. While Direct Preference Optimization (DPO) has shown promise for hallucination mitigation in earlier VLM architectures, its effectiveness on more recent models remains underexplored. This project investigates DPO-based hallucination mitigation on Qwen3-VL-4B-Instruct, examining the effects of DPO through training configuration ablations, preference data modality, and evaluation methodology. 

## Research Objectives
- Quantify hallucination rates in baseline VLMs using established benchmarks (POPE, MMHal-Bench).
- Fine-tune a VLM using DPO on a hallucination mitigation dataset (RLHF-V) and evaluate hallucination reduction.
- Analyse effects of DPO training using reward and log-probabilities plots and trade-offs between hallucination mitigation and informativeness.

## Setup
### Prerequisites
- Python 3.11+
- CUDA 11.8+ (for GPU training)

### Installation
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/HilariusJeremy/fyp-multimodal-hallucination.git
cd fyp-multimodal-hallucination

# If you already cloned without --recurse-submodules, run:
# git submodule update --init --recursive

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install LLaMA-Factory
pip install -e external/llama-factory[torch,metrics]

# Install lmms-eval
pip install -e external/lmms-eval
```

### Dataset
The preference data is loaded directly from HuggingFace: [openbmb/RLHF-V-Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset). It contains 5,733 preference pairs (chosen/rejected responses) across image-text tasks drawn from COCO, VQAv2, ShareGPT4V, and other sources.

Download it once to data/raw/ using the HuggingFace CLI:
```bash
huggingface-cli download openbmb/RLHF-V-Dataset --repo-type dataset --local-dir data/raw
```

### SFT Training
Run the conversion script to produce LLaMA-Factory-compatible files in data/processed/:
```bash
python scripts/convert_arrow_rlhf_v_to_sft.py data/raw/train.arrow
```

Run training using LLaMA-Factory:
```bash
cd external/llama-factory
llamafactory-cli train ../../configs/llama-factory/train/qwen3vl_lora_sft.yaml
```

Perform merging for inference:
```bash
llamafactory-cli export ../../configs/llama-factory/merge/qwen3vl_lora_sft.yaml
```




