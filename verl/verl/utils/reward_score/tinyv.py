import re
import re
import signal
from . import prime_math
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import requests
import json

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.
    
    Args:
        string: Input string containing LaTeX code
        
    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def format_score(solution_str, good_score=0., bad_score=-1.):
    # check if </think> is in the solution str, and only appear once
    if solution_str.count("</think>") != 1:
        score = bad_score
    else:
        score = good_score

    # cot is the text before first </think>
    cot = solution_str[:solution_str.find("</think>")]
    cot = cot.strip()
    # answer is the text after last </think>
    answer = solution_str[solution_str.rfind("</think>") + len("</think>"):]
    answer = answer.strip()
    return score, cot, answer


# get the tinyv config for the given model
def get_tinyv_config(model_name: str):
    with open('tinyv_config.json', 'r') as f:
        tinyv_config_all = json.load(f)

    if model_name not in tinyv_config_all:
        raise KeyError(f"Model '{model_name}' not found in tinyv_config. Available models: {list(tinyv_config_all.keys())}")
    return tinyv_config_all[model_name]


client = OpenAI(api_key='token-abc123', base_url='http://localhost:8000/v1')
# get the model name from the response
response = requests.get('http://localhost:8000/v1/models')
try:
    models_data = response.json()
    VERIFIER_MODEL_NAME = models_data['data'][0]['id']
    VERIFIER_MODEL_CONFIG = get_tinyv_config(VERIFIER_MODEL_NAME)
    TINYV_PROMPT = VERIFIER_MODEL_CONFIG['template']
    MAX_COMPLETION_TOKENS = VERIFIER_MODEL_CONFIG['max_completion_tokens']
    TEMPERATURE = VERIFIER_MODEL_CONFIG['temperature']
    TOP_P = VERIFIER_MODEL_CONFIG['top_p']
    IS_THINK_MODEL = VERIFIER_MODEL_CONFIG['is_think_model']
except Exception as e:
    raise Exception(f"Verifier LLM is not running. Please run the verifier first.")


def model_infer(msg, model, retry=3, temperature=0, top_p=1):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=MAX_COMPLETION_TOKENS
        )
        parsed_resp = resp.choices[0].message.content.strip()

        # For think verifiers, only consider the answer part
        if IS_THINK_MODEL:
            if "Answer:" in parsed_resp: # Qwen2.5
                parsed_resp = parsed_resp.split("Answer:")[1].strip()
            elif "</think>" in parsed_resp: # Qwen3
                parsed_resp = parsed_resp.split("</think>")[1].strip()
            else:
                print(f"Invalid response: {parsed_resp}")
                score = 0
                return score
        
        if 'true' in parsed_resp.lower():
            score = 1 
        elif 'false' in parsed_resp.lower():
            score = 0
        else:
            print(f"Invalid response: {parsed_resp}")
            score = 0
        return score

    except Exception as e:
        print(f"LLM Verifier InferenceError: {e}")
        if retry > 0:
            # In case the model is not working, we try again with a higher temperature
            return model_infer(msg, model, retry=retry-1, temperature=TEMPERATURE+0.3, top_p=1)
        else:
            return 0

def tinyv_score(question_str:str, ground_truth:str, model_answer:str, debug=False):
    global client
    msg = [
        {"role": "user", "content": TINYV_PROMPT.replace("{{QUESTION}}", question_str).replace("{{GROUND_TRUTH_ANSWER}}", ground_truth).replace("{{MODEL_ANSWER}}", model_answer)}
    ]
    if debug:
        print(f"TinyV Prompt: {msg}")
    
    tinyv_score = model_infer(msg, VERIFIER_MODEL_NAME, retry=3, temperature=TEMPERATURE, top_p=TOP_P)

    return tinyv_score


def _compute_score(solution_str:str, ground_truth:str, question_str:str, tinyv_setup=None, tinyv_weight=None, debug=False):
    prime_is_correct, prime_format_correctness, answer = prime_math.compute_score(solution_str, ground_truth)

    # Zhangchen: We don't consider format reward for now for consistency with prime
    if tinyv_setup == 'addon':
        if prime_is_correct == False:
            tinyv_reward = tinyv_score(question_str, ground_truth, answer) * tinyv_weight
        else:
            tinyv_reward = 1
        score = tinyv_reward
    elif tinyv_setup == 'tinyv_only':
        tinyv_reward = tinyv_score(question_str, ground_truth, answer)
        score = tinyv_reward
    elif tinyv_setup == 'mixed':
        tinyv_reward = tinyv_score(question_str, ground_truth, answer)
        score = tinyv_reward * tinyv_weight + prime_is_correct * (1 - tinyv_weight)
    else:
        raise ValueError(f"Invalid tinyv_setup: {tinyv_setup}")

    if debug:
        print(f"Question: {question_str}")
        print(f"TinyV Setup:             : {tinyv_setup}")
        print(f"Prime Correctness        : {prime_is_correct}")
        print(f"Prime Format Correctness : {prime_format_correctness}")
        print(f"Model Answer             : {answer}")
        print(f"Ground Truth Answer      : {ground_truth}")
        print(f"Assigned tinyv Reward    : {tinyv_reward}")
        print(f"Assigned score           : {score}")
        print("-"*80)

    return score


def compute_score(solution_str, ground_truth, extra_info=None, tinyv_setup=None, tinyv_weight=None, format_reward_max=0., tinyv_reward_max=1.):
    question_str = extra_info['question']

    # format_reward, tinyv_reward = _compute_score(solution_str, ground_truth, question_str)
    # score = format_reward + tinyv_reward
    # marker = "✅" if score == (format_reward_max + tinyv_reward_max) else "❌"

    score = _compute_score(solution_str, ground_truth, question_str, tinyv_setup, tinyv_weight)
    marker = "✅" if score == tinyv_reward_max else "❌"
    print(f"Reward: {score}. {marker}")

    return score