# Code Llama Fine-tuning to Fill in the Middle

🚧 This repository is under construction 🚧

This repository allows you to train Code Llama model to fill in the middle on your own dataset. Example training:

```bash
python train.py \
  --model_name_or_path "codellama/CodeLlama-7b-hf" \
  --dataset_name "<your-hf-dataset-name>" \
  --splits "train" \
  --max_seq_len 2048 \
  --max_steps 1000 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 5 \
  --log_level "info" \
  --logging_strategy "steps" \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --push_to_hub \
  --hub_private_repo False \
  --hub_strategy "every_save" \
  --bf16 True \
  --learning_rate 2e-4 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.1 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --output_dir "your-output-model-name" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --use_reentrant True \
  --dataset_text_field "content" \
  --test_size 0.1 \
  --fim_rate 0.9 \
  --fim_spm_rate 0.5 \
  --use_peft_lora True \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --lora_target_modules "all-linear" \
  --use_4bit_quantization True \
  --use_nested_quant True \
  --bnb_4bit_compute_dtype "bfloat16" \
  --use_flash_attn True
```

## Acknowledgements

This repository is based on https://github.com/pacman100/LLM-Workshop
