CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval.py \
--model_name_or_path "Qwen/Qwen2.5-32B-Instruct" \
--data_name "math" \
--prompt_type "qwen-instruct" \
--temperature 0.0 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--split "test" \
--max_tokens 32768 \
--seed 0 \
--top_p 1 \
--surround_with_messages \

