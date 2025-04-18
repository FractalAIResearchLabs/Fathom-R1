# 🧮 Ramanujan-Ganit-R1-14B-V1 - Recipe for Unlocking Math Reasoning at o4-mini level with 14B model under 16K context


<div align="center">

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Ramanujan--Ganit--R1--14B--V1-FFD21E?logo=huggingface&style=for-the-badge)](https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B-V1)

</div>

---

## 🧠 Overview

We introduce **Ramanujan-Ganit-R1-14B-V1** - a compact and compute-efficient 14B language model optimized for **concise and accurate mathematical reasoning**.  
Trained via a multi-stage pipeline combining **reinforcement learning**, **supervised fine-tuning**, and **curriculum learning**, it delivers **performance rivaling closed-source o4-mini (low)** — all while staying within a **16K context window** and a modest compute budget.

---

## 🧪 Why Another Math Model?

Recent advances like DeepSeek-Math and Light-R1 show the potential of large-scale instruction tuning and distillation. However, these approaches often rely on:

- Larger model scales (32B, 70B+)
- Expensive context lengths (32K+)
- Redundant or verbose reasoning

We address these constraints by focusing on **response efficiency** — training the model to generate only the **minimal reasoning needed** to reach a correct solution, while scaling curriculum difficulty gradually.

---

## 🏗️ Training Pipeline





## 📊 Benchmarking

Evaluated across **AIME25**, **HMMT25**, and **GPQA**, using `pass@1` and `cons@64` metrics — as defined in the DeepSeek-R1 and LIMO framework.

| Model                    | AIME25 (p@1 / c@64) | HMMT25 (p@1 / c@64) |
|--------------------------|---------------------|----------------------|
| o4-mini-low              | 60.2 / 76.67        | 39.11 / 53.33        |
| Light-R1-14B             | 51.15 / 76.67       | 34.11 / 50.00        |
| R1-Distill-14B           | 45.5 / 63.33        | 30.00 / 50.00        |
| **Ramanujan-Ganit‑V1**   | **51.88 / 76.67**   | **35.78 / 56.66**    |

📈 On both AIME and HMMT, **Ramanujan-Ganit-R1-14B-V1** is the **strongest open-source model** under 14B and 16K context.

---

## 🌍 Generalization Beyond Math

Notably, we also observe out-of-domain improvement in **GPQA**, even though there wasn't a single instance of science reasoning based questions in our training data. 
This indicates that training solely on mathematics-focused datasets potentially facilitates generalization across diverse domains, a finding similar to what Light-R1 had observed.
#### ✅ GPQA Benchmark Comparison
| **Model**         | **pass@1** | **cons@64** |
|-------------------|------------|-------------|
| LightR1‑14B       | 56.94      | 65.15       |
| R1‑distill‑14B    | 54.19      | 64.14       |
| R1‑distill‑32B    | 64.57      | 69.70       |
| R1‑670B           | 71.88      | 74.24       |
| Ramanujan-Ganit‑R1-14B-V0.4           | 56.35      | 66.67       |
| Ramanujan-Ganit‑R1-14B-V0.6           | 58.91      | 63.13       |
| **Ramanujan-Ganit‑R1-14B-V1**  | 59.13 | 66.16  |


## Ablation Study on Token Efficiency
To assess reasoning efficiency, we compare the **average response lengths** across  AIME25, and HMMT25. While models like **Light-R1-14B**,  **R1-distill‑14B** and **Ramanujan-Ganit‑R1-14B-V0.6** tend to generate longer chains, **Ramanujan-Ganit‑R1-14B-V1** consistently produces **more concise responses** without sacrificing performance. 
#### Average Response Length (Tokens)

| Model            | AIME25 | HMMT25 |
|------------------|--------|--------|
| Light-R1-14B         | 11330  | 12680  |
| R1-distill-14B   | 10878  | 12263  |
| Ramanujan-Ganit‑R1-14B-V0.4          | 10570  | 11950  |
| Ramanujan-Ganit‑R1-14B-V0.6         | 11236  | 12717  |
| **Ramanujan-Ganit‑R1-14B-V1**      | 10083  | 12100  |


## 📜 License

🟡 Released under **Apache 2.0** — free for commercial and research use.

---

## 📖 Citation

```bibtex
@misc{ramanujan14b2024,
  title={Ramanujan-Ganit-R1-14B},
  author={Fractal AI Research},
  year={2024},
  url={https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B-V1}
}
