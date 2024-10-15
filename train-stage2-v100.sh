export WANDB_API_KEY="d8d48d5c16ca9b51769d812605aed1929aca30e1"
export WANDB_PROJECT="OCR2.0"
export WANDB_NAME="POD-DocLayNet-wo-others-3e-unfreeze-backbone-stage2-pod-ocr-2"

deepspeed GOT-OCR-2.0-master/GOT/train/train_GOT.py \
  --deepspeed GOT-OCR-2.0-master/zero_config/zero2-v100.json \
  --model_name_or_path outputs/POD-DocLayNet-wo-others-3e-unfreeze-backbone \
  --use_im_start_end True \
  --fp16 True \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 1 \
  --weight_decay 0. \
  --warmup_ratio 0.001 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --report_to wandb \
  --per_device_train_batch_size 2 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --datasets pod-ocr-2 \
  --output_dir /blob/workstation/LLM-Finetuning/GOT-OCR-2.0/POD-DocLayNet-wo-others-3e-unfreeze-backbone-stage2-pod-ocr-2/ \
  --resume_from_checkpoint outputs/POD-DocLayNet-wo-others-3e-unfreeze-backbone-stage2-pod-ocr-2/checkpoint-7000