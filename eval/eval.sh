CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval.py \
--model_name_or_path "FractalAIResearch/Ramanujan-Ganit-R1-14B" \
--data_name "aime" \
--prompt_type "qwen-instruct" \
--temperature 0.0 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 64 \
--split "aime_2025" \
--max_tokens 16384 \
--seed 0 \
--top_p 1 \
--surround_with_messages \

