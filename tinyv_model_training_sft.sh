set -x
cd LLaMA-Factory

BASE_MODEL=${1:-"Qwen/Qwen2.5-1.5B-Instruct"}
DATASET=${2:-"zhangchenxu/TinyV_Training_Data_Balanced"}
HUB_TOKEN=${3:-""}
HUB_MODEL_ID=${4:-""}
LR=${5:-"1.0e-5"}
EPOCHS=${6:-"2"}
MAX_SAMPLES=${7:-"10000000"}

if [ -z "$HUB_TOKEN" ] || [ -z "$HUB_MODEL_ID" ]; then
  echo "WARNING: HUB_TOKEN or HUB_MODEL_ID is empty. Push to hub may fail."
fi

echo "DATASET: ${DATASET}"
echo "LR: ${LR}"
echo "EPOCHS: ${EPOCHS}"

BASE_MODEL_NAME=${BASE_MODEL#*/}
DATASET_NAME=${DATASET#*/}
OUTPUT_NAME=${BASE_MODEL_NAME}-SFT-${DATASET_NAME}-LR${LR}-EPOCHS${EPOCHS}
echo "OUTPUT_NAME: ${OUTPUT_NAME}"

llamafactory-cli train \
    --model_name_or_path ${BASE_MODEL} \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --dataset ${DATASET_NAME} \
    --template qwen \
    --cutoff_len 2048 \
    --max_samples ${MAX_SAMPLES} \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir ./models/${OUTPUT_NAME} \
    --logging_steps 1 \
    --save_steps 500 \
    --save_total_limit 1 \
    --plot_loss \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --report_to wandb \
    --run_name ${OUTPUT_NAME} \
    --push_to_hub True \
    --hub_strategy end \
    --hub_always_push False \
    --hub_model_id ${HUB_MODEL_ID} \
    --hub_token ${HUB_TOKEN} \
    --hub_private_repo True \

cd ..
