# LLM-TMP



## Llama-3.1-8B-Instruct
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/Llama-3.1-8B-Instruct \
    --dataset identity,TMP_Data_2025_2_25 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM_TMP/data \
    --template llama3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM_TMP/saves/LLaMA3.1-8B/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```


```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /home/sky-lab/codes/Llama-3.1-8B-Instruct \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM_TMP/saves/LLaMA3.1-8B/lora/sft  \
    --template llama3 \
    --finetuning_type lora
```

## DeepSeek-R1-Distill-Llama-8B

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/DeepSeek-R1-Distill-Llama-8B \
    --dataset identity,TMP_Data_2025_2_25 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM_TMP/data \
    --template deepseek3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM_TMP/saves/DeepSeek-R1-Distill-Llama-8B/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /home/sky-lab/codes/DeepSeek-R1-Distill-Llama-8B \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM_TMP/saves/DeepSeek-R1-Distill-Llama-8B  \
    --template deepseek3 \
    --finetuning_type lora
```

## Qwen2.5-VL-7B-Instruct


```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/Qwen2.5-VL-7B-Instruct \
    --dataset identity,TMP_Data_2025_2_25 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM_TMP/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM_TMP/saves/Qwen2.5-VL-7B-Instruct/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 50 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```