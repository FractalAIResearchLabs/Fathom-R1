# Ganit‑R1: An Adaptive Model for Reasoning Chain Compression and Decompression

## Overview

**Ganit‑R1** is a 14 billion‑parameter language model optimized for multi‑step chain‑of‑thought reasoning across very long contexts (up to 16 384 tokens). It achieves strong pass@1 and consistency@64 scores on challenging benchmarks, striking a practical balance between answer accuracy and reasoning coherence. This repository provides the pre‑trained model weights, evaluation scripts, and result visualizations to help you integrate and assess Ganit‑R1 in your own workflows.

## Datasets

### Dataset 1: RL Compression Data

We begin by sampling multiple responses for each question from the R1‑distill‑14B model solely to compute solve rates. Each chain is truncated at 6 000 tokens—if a solution requires more than this budget, that sample is marked incorrect. We then calculate the solve rate as the fraction of truncated chains that still arrive at the correct answer. Finally, we filter to those questions whose truncated solve rate lies between 0 and 0.5. This filtered set of problems drives the RL compression phase, guiding Ganit‑R1 to focus on questions that are challenging under a tight token budget.

### Dataset 2: SFT Shortest Chains

The second dataset focuses on **conciseness**. Again sampling multiple responses per question from the 14 billion‑parameter model, we identify which responses are correct and then pick the **shortest** chain of thought among them. These shortest‑correct examples form the training data for our supervised fine‑tuning stage, guiding Ganit‑R1 to elaborate its reasoning only as much as necessary when chain length is at a premium.

### Dataset 3: Curriculum‑Learning Data

For our curriculum‑learning set, we start with the R1‑distill‑14B checkpoint and annotate each question’s difficulty using the o3‑mini evaluator on the open‑R1 benchmark. We retain only those questions rated at difficulty 5 or above. Next, we recompute solve rates via multiple samplings from the 14 billion‑parameter model and filter to problems with solve rates between 0.2 and 0.6. By ordering this subset from easier to harder across training epochs, we enable a curriculum learning strategy that gradually exposes Ganit‑R1 to more challenging reasoning tasks.

## Training Strategy

1. **Model 1 (RL Compression)**  
   - **Base checkpoint**: R1‑distill‑14B  
   - **Data**: Dataset 1  
   - **Context length**: 6 000 tokens  
   - **Objective**: Reinforcement learning to compress reasoning chains without losing accuracy.  

   In this phase, Ganit‑R1 learns to identify and retain only the most critical inference steps. By rewarding shorter chains that still produce correct answers under a strict 6 000‑token budget, the model internalizes a preference for brevity. This compressed reasoning capability serves as a foundation for more efficient inference in resource‑constrained settings.

2. **Model 2 (SFT Decompression)**  
   - **Base checkpoint**: Model 1  
   - **Data**: Dataset 2  
   - **Objective**: Supervised fine‑tuning to decompress and elaborate reasoning chains, guided by the shortest correct examples.  

   We specifically use the **shortest** correct chains as training examples because they represent the minimal set of inference steps necessary for correctness. By learning from these efficient yet complete chains, Ganit‑R1 acquires the ability to add explanatory detail only where it is needed—avoiding redundant or tangential reasoning while ensuring that every critical step is articulated clearly.

3. **Model 3 (Curriculum SFT)**  
   - **Base checkpoint**: R1‑distill‑14B  
   - **Data**: Dataset 3  
   - **Method**: Curriculum learning SFT—each epoch progresses from easier to harder questions.  

   To bolster robustness on challenging problems, we employ a curriculum learning schedule. Starting with moderate‑difficulty questions and gradually increasing complexity, Ganit‑R1 develops a scaffolded understanding of multi‑step reasoning. This staged approach prevents early overfitting on hard examples and ensures steady performance gains across difficulty levels.


4. **Final Model (Ganit‑R1)**  
   - **Components merged**: Model 2 (SFT Decompression) and Model 3 (Curriculum SFT)  
   - **Purpose of merging**: Combining the strengths of both branches creates a versatile model that can generate concise, clear explanations when brevity is important (from Model 2) while maintaining strong performance on difficult reasoning tasks learned through the curriculum schedule (from Model 3). The resulting Ganit‑R1 seamlessly adapts its chain‑of‑thought style to the demands of each question and available context.


## Evaluation

We evaluate Ganit‑R1 using the same metrics and sampling configuration introduced in the DeepSeek‑R1 paper, namely **pass@1** and **cons@64**. However, our evaluation is conducted under a reduced context window of 16 384 tokens, compared to DeepSeek‑R1’s 32 768 tokens, to better reflect practical deployment constraints.

- **pass@1**: Measures the fraction of problems correctly solved in the first generated sample.
- **cons@64**: Assesses consistency by sampling 64 reasoning chains per question and computing the majority vote accuracy.

**Evaluation Configuration**:

- Temperature: 0.6  
- top_p: 0.95  
- Number of sampled chains: 64  
- Context window: 16,384 tokens  

This setup allows us to benchmark Ganit‑R1’s reasoning performance and stability under realistic memory and inference budgets, while maintaining compatibility with the DeepSeek‑R1 evaluation protocol.

We utilize the evaluation framework provided by the [LIMO](https://github.com/GAIR-NLP/LIMO) repository to run inference and compute metrics. The evaluation process involves executing the `eval.sh` script with the following parameters:


## Results

We evaluate **Ganit‑R1** and several baseline models across four benchmarks: **AIME24**, **AIME25**, **HMMT25**, and **GPQA**. For each, we report `pass@1` and `cons@64`, following the same evaluation configuration.

| Model            | AIME24        |              | AIME25        |              | HMMT25        |              |
|------------------|---------------|--------------|---------------|--------------|---------------|--------------|
|                  | pass@1        | cons@64      | pass@1        | cons@64      | pass@1        | cons@64      |
| LightR1‑14B      | 68.8          | 86.67        | 51.15         | 76.67        | 34.11         | 50.00        |
| R1‑distill‑14B   | 63.8          | 80.00        | 45.5          | 63.33        | 30.00         | 50.00        |
| R1‑distill‑32B   | 66.8          | 83.33        | 49.64         | 73.33        | 33.02         | 53.33        |
| R1‑670B          | 75.52         | 86.67        | 61.25         | 83.33        | 42.19         | 56.67        |
| **Ganit‑R1‑14B** | **68.13**     | **83.33**    | **51.88**     | **76.67**    | **35.78**     | **56.66**    |
| o1‑mini          | 60.05         | 80.00        | 50.71         | 63.33        | 35.15         | 46.67        |
| o3‑mini‑low      | 57.44         | 66.67        | 42.6          | 53.33        | 26.61         | 33.33        |
| o3‑mini‑medium   | 77.39         | 90.00        | 72.24         | 83.33        | 49.21         | 60.00        |
| o1‑preview       | 48.89         | 56.66        | 33.33         | 36.67        | 17.78         | 20.00        |
| gpt‑4.5‑preview  | 37.78         | 43.33        | 34.44         | 40.00        | 16.67         | 20.00        |
| Model 2          | 66.8          | 83.33        | 50.94         | 73.33        | 33.7          | 40.00        |
| Model 3          | 63.8          | 83.33        | 50.63         | 76.67        | 32.19         | 50.00        |


**Ganit‑R1** demonstrates competitive performance across all datasets, improving over the original distill checkpoints and closely matching or surpassing other strong baselines in several settings. Its consistency across diverse mathematical domains highlights its balanced reasoning ability.


<!-- ## Ablation Studies

We performed several ablations to isolate the contributions of each component:

1. **No RL Compression**: Skipping Model 1 reduced pass@1 by X%.  
2. **No Decompression SFT**: Skipping Model 2 hurt consistency@64 by Y%.  
3. **No Curriculum Learning**: Skipping the CL schedule in Model 3 led to Z% lower performance on hard questions.
4. **Merged model vs it's components** -->


