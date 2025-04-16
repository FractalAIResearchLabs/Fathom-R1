# Math Problem Evaluation Framework

This repository contains scripts for evaluating Large Language Models (LLMs) on mathematical reasoning tasks. The evaluation framework includes both inference (using the VLLM framework) and evaluation components.

## Environment Setup

When setting up the environment, pay attention to package version numbers, especially for those with specific version requirements noted in the documentation.

```bash
pip install -r requirements.txt
```

## Benchmark Evaluation

### Data Preparation

All benchmark datasets for evaluation should be placed in the `./data` directory.

To add a new test dataset, follow the format of existing benchmarks in the `./data` directory.

### Prompt Configuration

For mathematical problems, we use the Qwen-instruct template:

```python
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

few_shot_prompt = ""

question_format = """{question}"""
```

When adding a new mathematics benchmark, you can directly copy the above content to the corresponding `./prompts/qwen-instruct/xxx.py` file.

### Rule-Based Evaluation Interface

The framework provides a simple interface for rule-based evaluation of model predictions. Here's a basic example of how to use it:

```python
from utils.grader import check_is_correct
from utils.parser import extract_answer

def evaluate_prediction(model_pred: str, gold_answer: str) -> bool:
    """
    Evaluate a model's prediction against a gold answer.
    
    Args:
        model_pred (str): The model's prediction with answer in \boxed{}.
        gold_answer (str): The correct answer to compare against.
    
    Returns:
        bool: True if the prediction matches the gold answer, False otherwise.
    """
    # Extract the answer from model prediction
    extracted_answer = extract_answer(model_pred)
    
    # Check if the extracted answer matches the gold answer
    return check_is_correct(extracted_answer, gold_answer)

if __name__ == "__main__":
    # Example usage
    model_pred = "Let's solve this step by step:\n1. First...\n2. Then...\nSo the final answer is \\boxed{\\frac{1}{4}}"
    gold_answer = "0.25"
    
    is_correct = evaluate_prediction(model_pred, gold_answer)
    print(f"Prediction is correct: {is_correct}")  # True
```

The evaluation utilities handle various answer formats:
- Fractions (e.g., "\\frac{1}{4}")
- Decimals (e.g., "0.25")
- Mixed numbers
- Mathematical expressions

### Running Evaluation

Execute the evaluation script using:

```bash
bash eval.sh
```

Parameters in `eval.sh`:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval.py \
--model_name_or_path "/path/to/model/weights" \  # Path to model weights
--data_name "math" \  # Benchmark name (corresponding to first-level directory in ./data)
--prompt_type "qwen-instruct" \  # Default chat template
--temperature 0.0 \  # Sampling temperature
--start_idx 0 \  # Starting index for evaluation data
--end_idx -1 \  # Ending index for evaluation data
--n_sampling 64 \  # Number of samples per question
--split "test" \  # Benchmark subset partition
--max_tokens 16384 \  # Maximum output length
--seed 0 \  # Random seed
--top_p 1 \  # Top-p sampling parameter
--surround_with_messages \  # Enable this flag if using chat template
```

## Acknowledgments

Our evaluation code is modified from [LIMO](https://github.com/GAIR-NLP/LIMO/tree/main/eval). We thank their team for their valuable contributions to the community.