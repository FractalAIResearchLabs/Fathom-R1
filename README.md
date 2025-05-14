# 🧮 Ramanujan-Ganit-R1: $499 Training Recipe for Unlocking Math Reasoning at o4-mini level with just 14B parameters under 16K context

 
<div align="center">
 
[![collections](https://img.shields.io/badge/HFModels-Ramanujan--Ganit--R1--14B-yellow?logo=huggingface&style=for-the-badge)](https://huggingface.co/collections/FractalAIResearch/ramanujan-ganit-r1-models-681b41a149682c7e32f8a9f2) 
[![dataset](https://img.shields.io/badge/HFData-Ramanujan--Ganit--R1--Data-green?logo=huggingface&style=for-the-badge)](https://huggingface.co/collections/FractalAIResearch/ramanujan-ganit-r1-datasets-681b42fe6f20d4b11fc51d79) 
[![space](https://img.shields.io/badge/HFSpace-Ramanujan--Ganit--R1--14B-red?logo=huggingface&style=for-the-badge)](https://huggingface.co/spaces/FractalAIResearch/Ramanujan-Ganit-R1-14B)

</div>

<p align="center"> <img src="./images/image.png" style="width: 100%;" id="title-icon">       </p>

---

## Overview

Reasoning models often require high post-training budgets and extremely long reasoning chains(think 32k/64k) for maximising performance. Can we improve these models even if both these parameters are capped?

To this end, we first introduce: **Ramanujan-Ganit-R1-14B**, a 14-billion-parameter reasoning language model derived from Deepseek-R1-Distilled-Qwen-14B, trained at an affordable cost of only $499, and achieving SOTA mathematical reasoning performance within a 16K context window. On the latest olympiad level exams: AIME-25 and HMMT-25, our model not only **surpasses o3-mini-low, o1-mini and LightR1-14B (16k)** at pass@1 scores (averaged over 64 runs) but also delivers **performance rivaling closed-source o4-mini (low)** w.r.t cons@64 — all while staying within a **16K context window**. It achieves 52.71% Pass@1 accuracy on AIME2025 and 35.26% Pass@1 accuracy on HMMT25 (+7.2% and +5.2% improvement over the base model respectively). When provided with additional test-time compute in the form of cons@64, it achieves an impressive 76.7% accuracy on AIME2025 and 56.7% accuracy on HMMT25 (+13.4% and +6.7% improvement over the base model respectively). We perform supervised fine-tuning (SFT) on carefully curated datasets using a specific training approach, followed by model merging, achieving this performance at a total cost of just $499!

We also introduce **Ramanujan-Ganit-R1-14B-RS**, another model achieving performance comparable to our first, at a total training cost of just $967. It leverages post-training techniques—including reinforcement learning and supervised fine-tuning—in a multi-stage, cost-effective manner, followed by model merging.

We are **open-sourcing our models, training recipes and datasets** which we believe will help the community to progress further in the reasoning domain. 

---

## Release Assets

- Training Recipe Blog: [🤗 $499 training recipe for creating Ramanujan-Ganit-R1-14B](https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B)
- Final Merged Models: [🤗 Ramanujan-Ganit-R1-14B](https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B), [🤗 Ramanujan-Ganit-R1-14B-RS](https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B-RS)
- Intermediate Models:  [🤗 Ramanujan-Ganit-R1-14B-V0.6](https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B-V0.6), [🤗 Ramanujan-Ganit-R1-14B-V0.4](https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B-V0.4), [🤗 Ramanujan-Ganit-R1-14B-V0.4-RS](https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B-V0.4-RS)
- Ramanujan-Ganit-R1-14B Datasets: [🤗 V0.6-Iterative-Curriculum-Learning](https://huggingface.co/datasets/FractalAIResearch/Ramanujan-Ganit-V0.6-Iterative-Curriculum-Learning), [🤗 V0.4-SFT-Shortest-Chains](https://huggingface.co/datasets/FractalAIResearch/Ramanujan-Ganit-V0.4-SFT-Shortest-Chains), [🤗 V0.4-RL-Compression](https://huggingface.co/datasets/FractalAIResearch/Ramanujan-Ganit-V0.4-RL-Compression)

---

## 📊 Evaluation
We evaluate Ramanujan-Ganit‑R1-14B using the same metrics and sampling configuration introduced in the DeepSeek‑R1 paper, namely **pass@1** and **cons@64**. However, our evaluation is conducted under a reduced output budget of 16,384 tokens, compared to DeepSeek‑R1’s 32,768 tokens, to better reflect practical deployment constraints.

- **pass@1**: Pass@1 is computed as the average correctness over k sampled solution chains per problem (in our experiments we keep k=64).
- **cons@64**: Assesses consistency by sampling 64 reasoning chains per question and computing the majority vote accuracy.

**Evaluation Configuration**:

- Temperature: 0.6  
- top_p: 0.95  
- Number of sampled chains: 64  
- Context: 16,384 tokens  

This setup allows us to benchmark Ramanujan-Ganit-R1-14B’s reasoning performance and stability under realistic memory and inference budgets, while maintaining compatibility with the DeepSeek‑R1 evaluation protocol.

We utilize the evaluation framework provided by the [LIMO](https://github.com/GAIR-NLP/LIMO) repository to run inference and compute metrics.
For detailed instructions and implementation details, please refer to [`eval/README.md`](./eval/readme.md).

---

## Results
We evaluate and compare **Ramanujan-Ganit‑R1-14B** with several baseline models across 3 challenging benchmarks:  **AIME25**, **HMMT25**, and **GPQA**. For each, we report `pass@1` and `cons@64`, following the same evaluation configuration.


| Model            | AIME25         |               | HMMT25         |               |
|------------------|----------------|---------------|----------------|---------------|
|                  | pass@1         | cons@64       | pass@1         | cons@64       |
| **Closed-Source Models**               |                |               |                |               |
| o1‑mini          | 50.71          | 63.33         | 35.15          | 46.67         |
| o3‑mini‑low      | 42.60          | 53.33         | 26.61          | 33.33         |
| o3‑mini‑medium   | 72.24          | 83.33         | 49.21          | 60.00         |
| o4-mini-low      | 60.20          | 76.67         | 39.11          | 53.33         |
| o1‑preview       | 33.33          | 36.67         | 17.78          | 20.00         |
| gpt‑4.5‑preview  | 34.44          | 40.00         | 16.67          | 20.00         |
| **Open-Source Models**              |                |               |                |               |
| DeepSeek-R1-Distill-Qwen-14B        | 45.50          | 63.33         | 30.00          | 50.00         |
| DeepSeek-R1-Distill-Qwen-32B        | 49.64          | 73.33         | 33.02          | 53.33         |
| DeepSeekR1‑670B                     | 61.25          | 83.33         | 42.19          | 56.67         |
| LightR1‑14B                         | 51.15          | 76.67         | 33.75          | 50.00         |
| Ramanujan-Ganit‑R1-14B-RS (Ours)           | 52.03          | 76.67         | 35.00          | 53.33         |
| **Ramanujan-Ganit‑R1-14B (Ours)**          | **52.71**      | **76.67**     | **35.26**      | **56.67**     |



**Ramanujan-Ganit‑R1-14B** demonstrates highly competitive performance across all datasets, improving over the original R1-distilled models while closely matching or surpassing other strong baselines in several settings. 
On both AIME 25 and HMMT 25, our model shows the highest pass@1 as well as cons@64 scores among all the open-source models (including the bigger R1-Distilled-32B model), with R1-670B being the only exception.

In fact, we observe that Ramanujan-Ganit-R1-14B is superior to the first two generations of OpenAI's mini-reasoning models, including **o1-mini** and **o3-mini (low)** and it's performance closely matches that of newly released **o4-mini (low)** if provided with additional test-time compute.

---

## 🌍 Generalization Beyond Math: GPQA-Diamond

Notably, we also observe out-of-domain improvement in **GPQA-Diamond**, even though there wasn't a single instance of non-math questions in our training data. 
This indicates that our training methodology mentioned above and training on math questions facilitates generalization across diverse domains, a finding similar to what LightR1-14B & LIMO had observed.
#### ✅ GPQA Benchmark Comparison (16k)
| **Model**         | **pass@1** | **cons@64** |
|-------------------|------------|-------------|
| DeepSeek-R1-Distill-Qwen-14B          | 54.19      | 64.14       |
| LightR1‑14B                           | 56.94      | 65.15       |
| Ramanujan-Ganit‑R1-14B-RS             | 59.13      | 66.16       |
| **Ramanujan-Ganit‑R1-14B**            | **59.46**  | **66.16**   |

---

## Data Decontimination

Both benchmarks used (AIME 25 and HMMT 25) were released a few weeks after the release of the base model, ensuring no contamination occurred during the model's pre-training. The dataset corpora (Numina-Math 1.5 & OpenR1-Math) were released around the same time as these exams, with a cutoff date no later than 2024. Additionally, we conduct checks to verify there is no contamination in the training data.

## 📜 License

Our project is available under the MIT License, underscoring our dedication to open and inclusive AI innovation. By freely sharing our work, we aim to democratize AI technology, empowering researchers, developers, and enthusiasts everywhere to use, adapt, and expand upon it without limitation. This open and permissive approach promotes global collaboration, accelerates innovation, and enriches the AI community as a whole.

## Acknowledgments
We would like to acknowledge the following works for enabling our project:
- [Deepseek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)
- [OpenR1-Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [360-LLAMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory)
- [verl](https://github.com/volcengine/verl)
- [LIMO](https://github.com/GAIR-NLP/LIMO)
- [FuseAI](https://github.com/fanqiwan/FuseAI)

---

## 📖 Citation

```bibtex
@misc{ganit14b2025,
  title={Ramanujan-Ganit-R1: $499 Training Recipe for Unlocking Math Reasoning at o4-mini level with just 14B parameters under 16K context},
  author={Kunal Singh and Pradeep Moturi and Ankan Biswas and Siva Gollapalli and Sayandeep Bhowmick},
  howpublished={\url{https://huggingface.co/FractalAIResearch/Ramanujan-Ganit-R1-14B}},
  note={Hugging Face},
  year={2025}
}
```

---

## About Project Ramanujan

We initiated Project Ramanujan approximately one year ago, aiming to unlock intelligence and enhance AI agents by pushing the boundaries of advanced reasoning. Our key accomplishments include:
- ICLR'25 & NeurIPS'24-MATH-AI: [SBSC: Step-By-Step Coding for Improving Mathematical Olympiad Performance](https://arxiv.org/abs/2502.16666)
- Winners of HackerCupAI@NeurIPS'24 & ICLR'25-VerifAI: [Stress Testing Based Self-Consistency For Olympiad Programming](https://openreview.net/forum?id=7SlCSjhBsq)
- CVPR'25-MULA: [TRISHUL: Towards Region Identification and Screen Hierarchy Understanding for Large VLM based GUI Agents
](https://arxiv.org/abs/2502.08226))
- Silver Medal in AIMO'24



