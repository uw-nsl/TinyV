import os
import sys
import json
import warnings
import concurrent.futures
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import load_dataset_from_file, save_dataset
from utils import make_api_request_with_retry
from jinja2 import Environment, FileSystemLoader, exceptions
import re

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Unified Tagging Manager.")
    parser.add_argument("--tag_mission", type=str, default="gen_fn_analysis", help="The tagging mission.", choices=["gen_fn_analysis", "gen_synthetic", "gen_fn_analysis_synthetic"])
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Tag Model.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples per batch. Online <100, Offline <200.")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Save checkpoint every n batches")
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
checkpoint_every = args.checkpoint_every if args.tag_mission != "reward" else args.checkpoint_every*100
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
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('gen_fn_analysis.md').render(QUESTION=input['question'], GROUND_TRUTH_ANSWER=input['ground_truth'], MODEL_ANSWER=input['extracted_model_output'])
        return template
    elif mission == "gen_fn_analysis_synthetic":
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('gen_fn_analysis.md').render(QUESTION=input['question'], GROUND_TRUTH_ANSWER=input['ground_truth'], MODEL_ANSWER=input['synthetic_answer'])
        return template
    elif mission == "gen_synthetic":
        env = Environment(loader=FileSystemLoader('prompts'))
        template = env.get_template('gen_synthetic.md').render(QUESTION=input['question'], GROUND_TRUTH_ANSWER=input['ground_truth'])
        return template
    else:
        raise ValueError("Invalid mission. Available missions: quality, classification, type")

# Process item
def process_engine_responses(response, item, mission):
    if mission in ["gen_fn_analysis", "gen_fn_analysis_synthetic"]:
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
                        if mission in ["gen_fn_analysis", "gen_fn_analysis_synthetic"]:
                            item['llm_verifier_reasoning'] = None
                            item['llm_verifier_is_correct'] = None
                        return item
                        
                # Process successfully parsed JSON
                if mission in ["gen_fn_analysis", "gen_fn_analysis_synthetic"]:
                    item['llm_verifier_reasoning'] = tags_json.get("reasoning")
                    item['llm_verifier_is_correct'] = tags_json.get("is_correct")
            else:
                raise ValueError("No JSON object found in response")
        except Exception as e:
            print(f"[unitag.py] Failed to process item with error: {str(e)}")
            print(f"[unitag.py] Raw response from LLM tagger: {response}")
            if mission in ["gen_fn_analysis", "gen_fn_analysis_synthetic"]:
                item['llm_verifier_reasoning'] = None
                item['llm_verifier_is_correct'] = None

    elif mission == "gen_synthetic":
        try:
            # Extract all equivalent answers enclosed in <equivalent_answer_x>...</equivalent_answer_x> tags
            equivalent_answers = []
            pattern = r'<equivalent_answer_\d+>(.*?)</equivalent_answer_\d+>'
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                equivalent_answers.append(match.strip())
            item['equivalent_answers'] = equivalent_answers
        except Exception as e:
            print(f"[unitag.py] Failed to process item with error: {str(e)}")
            print(f"[unitag.py] Raw response from LLM tagger: {response}")
            item['equivalent_answers'] = None

    return item


# Process a batch of data using the API
def process_batch_with_api(batch, mission):
    if mission == "gen_synthetic":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_item = {
                executor.submit(
                    make_api_request_with_retry,
                    [
                        {'role': 'user', 'content': template_generator(item, mission)},
                        {'role': 'assistant', 'content': "```json\n{"}
                    ],
                    API_PARAMS,
                    API_ENDPOINT,
                    API_HEADERS
                ): item
                for item in batch
            }
            
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                api_response = future.result()
                # Append prefilling prefix
                api_response = "{" + api_response
                item = process_engine_responses(api_response, item, mission)
    else:
        raise ValueError("Invalid mission. Available missions: gen_synthetic")
                
    return batch

# Process a batch of data
def process_batch(batch, llm, params, mission, tokenizer=None):
    prompts = []
    for i, item in enumerate(batch):
        chat = [{"role": "user", "content": template_generator(item, mission)}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if mission == "gen_synthetic":
            template += "```json\n{"  # Prefilling for better generation
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        if mission == "gen_synthetic":
            model_response = "{" + outputs[i].outputs[0].text.strip() + "\n"
            # Remove additional information at the end of the response
            model_response = model_response[:model_response.rfind("}")+1]
        else:
            model_response = outputs[i].outputs[0].text.strip()
        item = process_engine_responses(model_response, item, mission)
    
    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, mission, llm, params, api, batch_size, tokenizer, checkpoint_file, checkpoint_every = 20):
    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"[unitag.py] Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        num_batches = (len(dataset) - last_checkpoint_idx + batch_size - 1) // batch_size
        print(f"[unitag.py] Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
        print(f"[unitag.py] Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size + last_checkpoint_idx
        end_idx = min((i + 1) * batch_size + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]

        if api:
            batch = process_batch_with_api(batch, mission)
        else:
            batch = process_batch(batch, llm, params, mission, tokenizer)

        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file every checkpoint_every batches
        if (i + 1) % checkpoint_every == 0:
            save_dataset(dataset[:end_idx], checkpoint_file)
            print(f"[unitag.py] Dataset checkpoint saved after batch {i + 1}.")

    return dataset


if __name__ == "__main__":
    input_file = args.input_file
    # Mission Settings
    if mission in ["gen_fn_analysis", "gen_fn_analysis_synthetic"]:
        output_file = f"{input_file[:input_file.rfind('.')]}_fn_analysis.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_fn_analysis_checkpoint.json"
    elif mission == "gen_synthetic":
        output_file = f"{input_file[:input_file.rfind('.')]}_synthetic.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_synthetic_checkpoint.json"
    else:
        raise ValueError("Invalid mission. Available missions: gen_fn_analysis, gen_fn_analysis_synthetic, gen_synthetic")
    # Change jsonl to json if args.save_as is json
    if args.save_as == "json":
        output_file = f"{output_file[:output_file.rfind('.')]}.json"

    # Load dataset
    if not args.debug:
        dataset = load_dataset_from_file(input_file)
    else:
        warnings.warn("Debug mode enabled. Only processing the first 100 samples.")
        dataset = load_dataset_from_file(input_file)[:100]

    if args.api:
        if args.tag_mission not in ["gen_fn_analysis", "gen_fn_analysis_synthetic"]:
            raise ValueError(f"The mission {args.tag_mission} is not supported by API.")

        print("[unitag.py] Start together API engine...")
        llm = None
        params = None
    else:
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
    updated_dataset = generate_and_update(dataset, mission, llm, params, args.api, batch_size, tokenizer, checkpoint_file, checkpoint_every)

    if args.save_as == "json":
        save_dataset(updated_dataset, output_file, convert_to_jsonl=False)
    else:
        save_dataset(updated_dataset, output_file, convert_to_jsonl=True)

    # Remove the checkpoint file after completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("[unitag.py] Final dataset saved. Checkpoint removed.")