MODEL_NAME=$1
GPU_ID=${2:-0}

if [ -z "$MODEL_NAME" ]; then
  echo "Please provide a model name as the first argument"
  exit 1
fi

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

# Generate a random port between 8000 and 10000
PORT=$((8000 + RANDOM % 2000))
echo -e "${BLUE}[LLM Verifier] Using port: $PORT${NC}"

# Start vllm in the background and save its PID
echo -e "${BLUE}[LLM Verifier] Initializing vllm server..."
CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve $MODEL_NAME \
        --tensor-parallel-size 1 \
        --port $PORT \
        --host 0.0.0.0 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.8 2>&1 | tee tinyv_vllm_output.log &
VLLM_PID=$!
echo -e "${BLUE}[LLM Verifier] VLLM server initialized with PID: $VLLM_PID ${NC}"

# Wait for the vllm server to start up
echo -e "${BLUE}[LLM Verifier] Waiting for VLLM server to be ready...${NC}"
MAX_RETRIES=60
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:$PORT/v1/models > /dev/null; then
        echo -e "${GREEN}[LLM Verifier] VLLM server is ready!${NC}"
        break
    else
        echo -e "${BLUE}[LLM Verifier] Waiting for VLLM server to start... ($(($RETRY_COUNT+1))/$MAX_RETRIES)${NC}"
        sleep 15
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\033[0;31m[LLM Verifier] Failed to start VLLM server after $MAX_RETRIES attempts. Exiting.${NC}"
    exit 1
fi

MODEL_NICKNAME=${MODEL_NAME##*/}
echo "Running TinyV on ${MODEL_NICKNAME}"

# Run the Python script
python tinyv_v1.py --model_name $MODEL_NAME --port $PORT > "${MODEL_NICKNAME}.txt" 2>&1

# Kill the vllm process
kill $VLLM_PID