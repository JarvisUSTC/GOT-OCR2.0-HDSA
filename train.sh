export WANDB_API_KEY="d8d48d5c16ca9b51769d812605aed1929aca30e1"

deepspeed   GOT-OCR-2.0-master/GOT/train/train_GOT.py \
 --deepspeed GOT-OCR-2.0-master/zero_config/zero2.json    --model_name_or_path GOT_weights/ \
 --use_im_start_end True   \
 --bf16 True   \
 --gradient_accumulation_steps 2    \
 --evaluation_strategy "no"   \
 --save_strategy "steps"  \
 --save_steps 500   \
 --save_total_limit 1   \
 --weight_decay 0.    \
 --warmup_ratio 0.001     \
 --lr_scheduler_type "cosine"    \
 --logging_steps 1    \
 --tf32 True     \
 --model_max_length 8192    \
 --gradient_checkpointing True   \
 --dataloader_num_workers 16    \
 --report_to wandb  \
 --per_device_train_batch_size 4    \
 --num_train_epochs 3  \
 --learning_rate 2e-5   \
 --datasets pod \
 --output_dir /blob/workstation/LLM-Finetuning/GOT-OCR-2.0/POD-DocLayNet-wo-others-3e-unfreeze-backbone/ \