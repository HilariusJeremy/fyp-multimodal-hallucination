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

### Dataset
The preference data is loaded directly from HuggingFace: [openbmb/RLHF-V-Dataset](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset). It contains 5,733 preference pairs (chosen/rejected responses) across image-text tasks drawn from COCO, VQAv2, ShareGPT4V, and other sources. No manual download is required as the dataset is cached automatically to ~/.cache/huggingface/datasets/ on first use.



