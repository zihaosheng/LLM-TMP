# LLM-TMP

## fine tune log
### 2025/03/07


## Llama-3.1-8B-Instruct
### Fine-tune
```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/Llama-3.1-8B-Instruct \
    --dataset identity,TMP_Data_2025_2_25_v1-1 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template llama3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/LLaMA3.1-8B_v1-1/lora/sft_2025-03-07-14-55-12 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
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
    --num_train_epochs 10.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```
```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
    --deepspeed /home/sky-lab/codes/LLaMA-Factory/examples/deepspeed/ds_z3_offload_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/Llama-3.1-8B-Instruct \
    --dataset identity,TMP_Data_2025_2_25 \
    --flash_attn auto \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template llama3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/LLaMA3.1-8B/lora/sft_2025-03-02-22-13-51 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 25000 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
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
    --num_train_epochs 10.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --bf16 True \
    --quantization_bit 4 \
    --quantization_method bitsandbytes \
    --double_quantization True
```
### Web chat
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /home/sky-lab/codes/Llama-3.1-8B-Instruct \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM-TMP/saves/LLaMA3.1-8B/lora/sft  \
    --template llama3 \
    --finetuning_type lora
```
### Batch eval
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/sky-lab/codes/Llama-3.1-8B-Instruct \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM-TMP/saves/LLaMA3.1-8B_v1-1/lora/sft_2025-03-07-14-55-12  \
    --eval_dataset TMP_Data_2025_2_25_v1-1 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template llama3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/LLaMA3.1-8B_v1-1/lora/predict_2025-03-07-14-55-12 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 40 \
    --predict_with_generate
```

## DeepSeek-R1-Distill-Llama-8B
### Fine-tune
```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/DeepSeek-R1-Distill-Llama-8B \
    --dataset TMP_Data_2025_2_25_v1-1 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template deepseek3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/DeepSeek-R1-Distill-Llama-8B_v1-1/lora/sft__2025-03-13-13-40-12 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
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
    --num_train_epochs 10.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```
```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/DeepSeek-R1-Distill-Llama-8B \
    --dataset alan_train_data \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template deepseek3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/DeepSeek-R1-Distill-Llama-8B_v1-2/lora/sft_2025-04-13 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
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
    --num_train_epochs 10.0 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```
### Web chat
```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli webchat \
    --model_name_or_path /home/sky-lab/codes/DeepSeek-R1-Distill-Llama-8B \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM-TMP/saves/DeepSeek-R1-Distill-Llama-8B_v1-1/lora/sft__2025-03-13-13-40-12/  \
    --template deepseek3 \
    --finetuning_type lora
```
### Batch eval
```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/sky-lab/codes/DeepSeek-R1-Distill-Llama-8B \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM-TMP/saves/DeepSeek-R1-Distill-Llama-8B_v1-1/lora/sft__2025-03-13-13-40-12/  \
    --eval_dataset TMP_Data_2025_2_25_v1-1 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template deepseek3 \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/DeepSeek-R1-Distill-Llama-8B_v1-1/lora/predict__2025-03-13-13-40-12/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 40 \
    --predict_with_generate
```

## Qwen/Qwen2.5-7B-Instruct-1M

```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /home/sky-lab/codes/Qwen2.5-7B-Instruct-1M \
    --dataset TMP_Data_2025_2_25_v1-1 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/Qwen2.5-7B-Instruct-1M_v1-1/lora/sft_2025-03-12-10-10-12 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
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
    --num_train_epochs 10.0 \
    --max_samples 1000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
```


```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /home/sky-lab/codes/Qwen2.5-7B-Instruct-1M \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM-TMP/saves/Qwen2.5-7B-Instruct-1M/lora/sft  \
    --template qwen \
    --finetuning_type lora
```

```bash
# idea 1 - implementation 2
CUDA_VISIBLE_DEVICES=0 llamafactory-cli webchat \
    --model_name_or_path /home/sky-lab/codes/Qwen2.5-7B-Instruct-1M/ \
    --adapter_name_or_path /home/sky-lab/Alan/TMP_LLM/saves/qwen/lora/sft_2025-04-04 \
    --template qwen \
    --finetuning_type lora
```
### Batch eval
```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/sky-lab/codes/Qwen2.5-7B-Instruct-1M \
    --adapter_name_or_path /home/sky-lab/SHENG_code/LLM-TMP/saves/Qwen2.5-7B-Instruct-1M_v1-1/lora/sft_2025-03-12-10-10-12/  \
    --eval_dataset TMP_Data_2025_2_25_v1-1 \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir /home/sky-lab/SHENG_code/LLM-TMP/saves/Qwen2.5-7B-Instruct-1M_v1-1/lora/predict_2025-03-12-10-10-12/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 40 \
    --predict_with_generate
```

```bash
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/sky-lab/codes/Qwen2.5-7B-Instruct-1M \
    --adapter_name_or_path /home/sky-lab/Alan/TMP_LLM/saves/qwen/lora/sft_2025-04-04 \
    --eval_dataset alan_train_data \
    --dataset_dir /home/sky-lab/SHENG_code/LLM-TMP/data \
    --template qwen \
    --finetuning_type lora \
    --output_dir /home/sky-lab/Alan/TMP_LLM/saves/qwen/lora/predict_2025-04-04-2 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 100 \
    --predict_with_generate
```