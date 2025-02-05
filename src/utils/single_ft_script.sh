#!/bin/bash
PREFIX=$1
LANGUAGE=$2
NUM_EPOCHS=$3
BATCH_SIZE_TRAINING=$4
LEARNING_RATE_POW=$5
DATA_LOC=$6
LEARNING_RATE_BASE=1
LEARNING_RATE=$(echo "scale=10; $LEARNING_RATE_BASE/10^$LEARNING_RATE_POW" | bc)
DIST_CKPT_ROOT_FOLDER="./model/"
DIST_CKPT_FOLDER="${PREFIX}/${LANGUAGE}/${NUM_EPOCHS}_${BATCH_SIZE_TRAINING}_${LEARNING_RATE_POW}"
OUT_DIR="${DIST_CKPT_ROOT_FOLDER}/${DIST_CKPT_FOLDER}"
DATASET="custom_dataset"
DATASET_FILE="./src/utils/load_dataset.py"

python -m llama_recipes.finetuning \
    --model_name './model/llama' \
    --use_peft True \
    --peft_method lora \
    --output_dir $OUT_DIR \
    --low_cpu_fsdp False \
    --run_validation True \
    --batch_size_training $BATCH_SIZE_TRAINING \
    --context_length 128 \
    --gradient_accumulation_steps 1 \
    --gradient_clipping False \
    --gradient_clipping_threshold 1.0 \
    --num_epochs $NUM_EPOCHS \
    --max_train_step 0 \
    --max_eval_step 0 \
    --num_workers_dataloader 8 \
    --lr $LEARNING_RATE \
    --weight_decay 0.0 \
    --seed 42 \
    --use_fp16 False \
    --mixed_precision False \
    --val_batch_size $BATCH_SIZE_TRAINING \
    --dataset $DATASET \
    --custom_dataset.file $DATASET_FILE \
    --custom_dataset.data_path $DATA_LOC \
    --quantization False \
    --save_model True \
    --save_optimizer False \
    --use_fast_kernels False \
    --use_wandb False \
    --lora_config.r 32 \
    --lora_config.lora_alpha 32