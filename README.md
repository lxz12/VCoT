# V-CoT
<p>
    <a href='#' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='#' target="_blank"><img src='https://img.shields.io/badge/Code-Partial-yellow'></a>
</p>

This is the implementation of the paper **V-CoT:  Collaborative Multi-Role Approach to Visual Reasoning**

---


## Architecture
![系统架构图](./frame/frame.pdf)

## Large Multimodal Models

GPT-4V: https://openai.com/research/gpt-4v-system-card

LLaVA-1.5-7B & LLaVA-1.5-13B: https://github.com/haotian-liu/LLaVA

SPHINX: https://github.com/sphinx-doc/sphinx

Qwen-VL-Chat: https://github.com/QwenLM/Qwen-VL

---

## Datasets

The evaluation framework supports 5 benchmark datasets:

| Dataset | Task Type | Samples | Metrics |
|---------|-----------|---------|---------|
| **Winoground** | Compositional Understanding | ~400 | Text/Image/Group Accuracy |
| **WHOOPS!** | Commonsense Reasoning | ~500 | Accuracy |
| **SEED-Bench** | Multi-dimension Understanding | ~14K | Category-wise Accuracy |
| **MMBench** | Systematic Multimodal Evaluation | ~3K | Category-wise Accuracy |
| **LLaVA-Bench-Wild** | Open-ended VQA | ~60 | Quality Scores |
---

