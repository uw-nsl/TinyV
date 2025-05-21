import os
import sys
import json
import warnings
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_dataset_from_file, save_dataset
from jinja2 import Environment, FileSystemLoader, exceptions
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocess.verl_reward_score.prime_math import match_answer

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Unified Tagging Manager.")
    parser.add_argument("--tag_mission", type=str, default="gen_fn_analysis", help="The tagging mission.", choices=["gen_fn_analysis"])
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Tag Model.")
    parser.add_argument("--input_folder", type=str, default=None, help="Input dataset folder name")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples per batch. Online <100, Offline <200.")
    # Remove checkpoint_every argument
    parser.add_argument("--api", type=bool, default=False, help="Use API to generate responses")
    parser.add_argument("--offline", action="store_false", dest="api", help="Use local vllm engine")
    parser.add_argument("--online", action="store_true", dest="api", help="Use Together API engine")
    parser.add_argument("--api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="API URL")
    parser.add_argument("--api_key", type=str, default=None, help="Together API Key")
    parser.add_argument("--debug", action="store_true", help="Debug mode. Only process the first 100 samples.")
    parser.add_argument("--save_as", type=str, default="json", choices=["json", "jsonl"], help="Save the generated responses as a what kind of file")

    # vllm Configs
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--quantization", type=str, default=None, choices=["fp8", "awq", "gptq", None])
    parser.add_argument("--kv_cache_dtype", type=str, default="auto", choices=["auto", "fp8"])
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    
    # Tagging Generation Configs
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()

args = get_args()
print(f"[unitag.py] Unified Tagging Manager. Arguments: {args}") # For logging

MODEL_NAME = args.model_path
batch_size = args.batch_size
mission = args.tag_mission

# Constants for the API
API_ENDPOINT = args.api_url
API_HEADERS = {
    "Authorization": args.api_key,
}
API_PARAMS = {
    "model": args.model_path,
    "max_tokens": args.max_tokens,
    "temperature": args.temperature,
    "repetition_penalty": args.repetition_penalty,
    "stop": ["}\n```", "<|im_end|>", "<|endoftext|>"]
}

def template_generator(input, mission):
    if mission == "gen_fn_analysis":
        question = input['extra_info']['question']
        ground_truth = input['ground_truth']
        is_matched, model_output = match_answer(input['response'])
        # print(f"[unitag.py] Ground truth: {ground_truth}")
        # print(f"[unitag.py] Model output: {model_output}")
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('gen_fn_analysis.md').render(QUESTION=question, GROUND_TRUTH_ANSWER=ground_truth, MODEL_ANSWER=model_output)
        return template
    else:
        raise ValueError("Invalid mission. Available missions: gen_fn_analysis")

# Process item
def process_engine_responses(response, item, mission):
    if mission == "gen_fn_analysis":
        try:
            # Try to extract just the JSON part using regex
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Handle invalid escape characters - more robust approach
                try:
                    # First try direct parsing
                    tags_json = json.loads(json_str)
                except json.JSONDecodeError:
                    # If that fails, try to fix invalid escape sequences
                    # Replace backslashes that aren't part of valid escape sequences
                    json_str = re.sub(r'\\([^"\\/bfnrtu]|u(?![0-9a-fA-F]{4}))', r'\\\\\1', json_str)
                    # Handle case where backslash is at the end of the string
                    json_str = re.sub(r'\\$', r'\\\\', json_str)
                    try:
                        tags_json = json.loads(json_str)
                    except json.JSONDecodeError as json_err:
                        print(f"[unitag.py] JSON decode error: {str(json_err)}")
                        print(f"[unitag.py] Problematic JSON string: {json_str}")
                        item['llm_verifier_reasoning'] = None
                        item['llm_verifier_is_correct'] = None
                        return item
                        
                # Process successfully parsed JSON
                item['llm_verifier_reasoning'] = tags_json.get("reasoning")
                item['llm_verifier_is_correct'] = tags_json.get("is_correct")
            else:
                raise ValueError("No JSON object found in response")
        except Exception as e:
            print(f"[unitag.py] Failed to process item with error: {str(e)}")
            print(f"[unitag.py] Raw response from LLM tagger: {response}")
            item['llm_verifier_reasoning'] = None
            item['llm_verifier_is_correct'] = None

    return item

# Process a batch of data
def process_batch(batch, llm, params, mission, tokenizer=None):
    prompts = []
    for i, item in enumerate(batch):
        chat = [{"role": "user", "content": template_generator(item, mission)}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        model_response = outputs[i].outputs[0].text.strip()
        item = process_engine_responses(model_response, item, mission)
    
    return batch

# Generate outputs, update dataset in batches (no checkpoint)
def generate_and_update(dataset, mission, llm, params, api, batch_size, tokenizer):
    num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
    print(f"[unitag.py] Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]

        if api:
            raise ValueError("API mode is not supported for this mission.")
        else:
            batch = process_batch(batch, llm, params, mission, tokenizer)

        dataset[start_idx:end_idx] = batch

    return dataset


if __name__ == "__main__":
    import glob

    input_folder = args.input_folder
    # Mission Settings
    if mission == "gen_fn_analysis":
        pass
    else:
        raise ValueError("Invalid mission. Available missions: gen_fn_analysis")

    # Find all train_step_x.json files in the folder, sorted by x
    file_list = glob.glob(os.path.join(input_folder, "train_step_*.json"))
    # Sort by the numeric x in train_step_x.json
    def extract_step_num(f):
        import re
        m = re.search(r"train_step_(\d+)\.json$", f)
        return int(m.group(1)) if m else -1
    file_list = sorted(file_list, key=extract_step_num)

    if len(file_list) == 0:
        raise ValueError(f"No train_step_x.json files found in {input_folder}")

    if args.api:
        raise ValueError("API mode is not supported for this mission.")

    print("[unitag.py] Start Local vllm engine...")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    llm = LLM(model=MODEL_NAME,
                dtype=args.dtype,
                quantization=args.quantization,
                kv_cache_dtype=args.kv_cache_dtype,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                trust_remote_code=True,
                enable_prefix_caching=True,)
    
    params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                stop=["}\n```", "<|im_end|>", "<|endoftext|>"],
                include_stop_str_in_output=True,
                )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    for input_file in tqdm(file_list):
        # Output file name
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(input_folder, f"{base_name}_fn_analysis.jsonl")
        if args.save_as == "json":
            output_file = output_file[:output_file.rfind('.')] + ".json"

        # Load dataset
        if not args.debug:
            dataset = load_dataset_from_file(input_file)
        else:
            warnings.warn("Debug mode enabled. Only processing the first 100 samples.")
            dataset = load_dataset_from_file(input_file)[:100]

        updated_dataset = generate_and_update(dataset, mission, llm, params, args.api, batch_size, tokenizer)

        if args.save_as == "json":
            save_dataset(updated_dataset, output_file, convert_to_jsonl=False)
        else:
            save_dataset(updated_dataset, output_file, convert_to_jsonl=True)