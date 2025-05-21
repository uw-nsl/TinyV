import re
import re
import signal
from . import prime_math
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import requests
import json

TINYV_PROMPT = """
You are an AI tasked with identifying false negatives in answer verification. A false negative occurs when a model's answer is essentially correct but is marked as incorrect due to minor discrepancies or formatting issues. Your job is to analyze the given question, ground truth answer, and model answer to determine if the model's answer is actually correct despite appearing different from the ground truth.

<question>{{QUESTION}}</question>

<ground_truth_answer>{{GROUND_TRUTH_ANSWER}}</ground_truth_answer>

<model_answer>{{MODEL_ANSWER}}</model_answer>

Return "True" if the model's answer is correct, otherwise return "False".
"""


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


client = OpenAI(api_key='token-abc123', base_url='http://localhost:8000/v1')
# get the model name from the response
response = requests.get('http://localhost:8000/v1/models')
try:
    models_data = response.json()
    VERIFIER_MODEL_NAME = models_data['data'][0]['id']
except Exception as e:
    raise Exception(f"Verifier LLM is not running. Please run the verifier first.")

def model_infer(msg, model, retry=3, temperature=0):
    if "think" in model.lower():
        max_completion_tokens = 2048
    else: # Non-think verifiers
        max_completion_tokens = 2
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens
        )
        parsed_resp = resp.choices[0].message.content.strip()

        # For think verifiers, only consider the answer part
        if "think" in model.lower():
            if "Answer:" in parsed_resp:
                parsed_resp = parsed_resp.split("Answer:")[1].strip()
            else:
                print(f"Invalid response: {parsed_resp}")
                score = 0
                return score
        
        if 'True' in parsed_resp:
            score = 1 
        elif 'False' in parsed_resp:
            score = 0
        else:
            print(f"Invalid response: {parsed_resp}")
            score = 0
        return score

    except Exception as e:
        print(f"LLM Verifier InferenceError: {e}")
        if retry > 0:
            return model_infer(msg, model, retry=retry-1, temperature=0.7)
        else:
            return 0

def tinyv_score(question_str:str, ground_truth:str, model_answer:str, debug=False):
    global client
    msg = [
        {"role": "user", "content": TINYV_PROMPT.replace("{{QUESTION}}", question_str).replace("{{GROUND_TRUTH_ANSWER}}", ground_truth).replace("{{MODEL_ANSWER}}", model_answer)}
    ]
    if debug:
        print(f"TinyV Prompt: {msg}")
    
    tinyv_score = model_infer(msg, VERIFIER_MODEL_NAME, retry=3, temperature=0)

    return tinyv_score


def _compute_score(solution_str:str, ground_truth:str, question_str:str, llm_verifier_setup=None, debug=False):
    prime_is_correct, prime_format_correctness, answer = prime_math.compute_score(solution_str, ground_truth)

    # Zhangchen: We don't consider format reward for now for consistency with prime

    if llm_verifier_setup == 'addon':
        if prime_is_correct == False:
            tinyv_reward = tinyv_score(question_str, ground_truth, answer)
        else:
            tinyv_reward = 1
    elif llm_verifier_setup == 'tinyv_only':
        tinyv_reward = tinyv_score(question_str, ground_truth, answer)
    else:
        raise ValueError(f"Invalid llm_verifier_setup: {llm_verifier_setup}")

    if debug:
        print(f"Question: {question_str}")
        print(f"LLM Verifier Setup:      : {llm_verifier_setup}")
        print(f"Prime Correctness        : {prime_is_correct}")
        print(f"Prime Format Correctness : {prime_format_correctness}")
        print(f"Model Answer             : {answer}")
        print(f"Ground Truth Answer      : {ground_truth}")
        print(f"Assigned tinyv Reward       : {tinyv_reward}")
        print("-"*80)

    return tinyv_reward


def compute_score(solution_str, ground_truth, extra_info=None, llm_verifier_setup=None, format_reward_max=0., tinyv_reward_max=1.):
    question_str = extra_info['question']

    # format_reward, tinyv_reward = _compute_score(solution_str, ground_truth, question_str)
    # score = format_reward + tinyv_reward
    # marker = "✅" if score == (format_reward_max + tinyv_reward_max) else "❌"

    tinyv_reward = _compute_score(solution_str, ground_truth, question_str, llm_verifier_setup)
    score = tinyv_reward
    marker = "✅" if score == tinyv_reward_max else "❌"
    print(f"Reward: {score}. {marker}")

    return score