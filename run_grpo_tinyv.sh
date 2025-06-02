#!/bin/bash
# The config is optimized for 8xA100
set -x
cd ./verl

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------------------------
# Cleanup function to kill the vllm server process
cleanup() {
    echo "Cleaning up background processes..."
    # Kill the vllm server process
    jobs -p | xargs -r kill
    
    exit 0
}

# Set trap to call cleanup function on script exit, interrupt, or termination
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------------------------
# Get the parameters from the command line
BASE_MODEL=${1:-"Qwen/Qwen2.5-7B"}
MAX_EPOCHS=${2:-"12"}
DATASET=${3:-"bigmath_rl_tinyv"}
NO_ENTROPY_LOSS_AND_KL=${4:-"False"}
VERIFIER_MODEL=${5:-"zhangchenxu/TinyV-1.5B"}
VERIFIER_SETUP=${6:-"addon"}

# MAIN CONFIG
MODEL_PATH=$BASE_MODEL
ROLLOUT_BATCH_SIZE=128
ROLLOUT_N_SAMPLE=8
PPO_MINI_BATCH_SIZE=64

MODEL_NICKNAME=$(echo $MODEL_PATH | cut -d'/' -f2)
VERIFIER_MODEL_NICKNAME=$(echo $VERIFIER_MODEL | cut -d'/' -f2)
TIMESTAMP=$(date +%Y%m%d%H%M%S)
if [[ "$DATASET" == *_tinyv* ]]; then
    RUN_NAME=${MODEL_NICKNAME}-${DATASET}-${VERIFIER_MODEL_NICKNAME}-${VERIFIER_SETUP}-3
else
    RUN_NAME=${MODEL_NICKNAME}-${DATASET}-3
fi

if [ "$NO_ENTROPY_LOSS_AND_KL" == "True" ]; then
    RUN_NAME=${RUN_NAME}-NEKL
    USE_KL_LOSS=FALSE
    USE_KL_IN_REWARD=FALSE
    KL_COEF=0.000
else
    USE_KL_LOSS=True
    USE_KL_IN_REWARD=True
    KL_COEF=0.001
fi

export RUN_NAME=${RUN_NAME}
export GLOBAL_RUN_NAME=${RUN_NAME}

TOTAL_SAMPLES=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N_SAMPLE )) # Number of experiences
echo "Number of experiences: $TOTAL_SAMPLES"

SAVED_DIR="./models_rl/${RUN_NAME}"
MAX_PROMPT_LEN=1024
MAX_RESPONSE_LEN=4096
PPO_MAX_TOKEN_LEN_PER_GPU=$(( 3 * $(( $MAX_PROMPT_LEN + $MAX_RESPONSE_LEN )) ))
MAX_NUM_BATCHED_TOKENS=$(( 1 * $(( $MAX_PROMPT_LEN + $MAX_RESPONSE_LEN )) ))

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

# ------------------------------------------------------------------------------------------------

# Initiate vllm server as a background process
echo -e "${BLUE}[LLM Verifier] Initializing vllm server..."
vllm serve $VERIFIER_MODEL \
        --tensor-parallel-size 4 \
        --port 8000 \
        --host 0.0.0.0 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.15 2>&1 | tee tinyv_vllm_output.log &
VLLM_PID=$!
echo -e "${BLUE}[LLM Verifier] VLLM server initialized with PID: $VLLM_PID ${NC}"

# Wait for the vllm server to start up
echo -e "${BLUE}[LLM Verifier] Waiting for VLLM server to be ready...${NC}"
MAX_RETRIES=50
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo -e "${GREEN}[LLM Verifier] VLLM server is ready!${NC}"
        break
    else
        echo -e "${BLUE}[LLM Verifier] Waiting for VLLM server to start... ($(($RETRY_COUNT+1))/$MAX_RETRIES)${NC}"
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\033[0;31m[LLM Verifier] Failed to start VLLM server after $MAX_RETRIES attempts. Exiting.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------------------------
# Run the GRPO training
echo -e "${BLUE}[GRPO Trainer] Running GRPO training...${NC}"

# export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
    data.train_files=./data/$DATASET/train.parquet \
    data.val_files="['./data/math_eval/math_benchmarks_aammo.parquet', './data/hardverify_math/test.parquet']" \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='TinyV-Test' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVED_DIR \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=64 \
    trainer.test_freq=8 \
    trainer.resume_mode=auto \
    trainer.total_epochs=$MAX_EPOCHS \
    reward_model.reward_manager=prime \
    reward_model.tinyv_setup=$VERIFIER_SETUP 2>&1 | tee ${RUN_NAME}.log
