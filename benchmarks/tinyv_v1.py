from openai import OpenAI
import requests
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="zhangchenxu/TinyV-1.5B")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--benchmark_path", type=str, default="/home/nsl/zhangchen/TinyV/benchmarks/HardVerify-Math.json")
args = parser.parse_args()

client = OpenAI(api_key='token-abc123', base_url=f'http://localhost:{args.port}/v1')
benchmarks = json.load(open(args.benchmark_path))
VERIFIER_MODEL_NAME = args.model_name

fn_correct = 0
tn_correct = 0
results = []

for entry in benchmarks:
    Question = entry["question"]
    Ground_Truth_Answer = entry["ground_truth"]
    Model_Answer_fn = entry["fn_output"]
    Model_Answer_tn = entry["tn_output"]

    PROMPT = """
    You are an AI tasked with identifying false negatives in answer verification. A false negative occurs when a model's answer is essentially correct but is marked as incorrect due to minor discrepancies or formatting issues. Your job is to analyze the given question, ground truth answer, and model answer to determine if the model's answer is actually correct despite appearing different from the ground truth. For numerical answers, the error margin is 5%.

    <question>{{QUESTION}}</question>

    <ground_truth_answer>{{GROUND_TRUTH_ANSWER}}</ground_truth_answer>

    <model_answer>{{MODEL_ANSWER}}</model_answer>

    Return "True" if the model's answer is correct, otherwise return "False".
    """

    fn_prompt = PROMPT.replace("{{QUESTION}}", Question).replace("{{GROUND_TRUTH_ANSWER}}", Ground_Truth_Answer).replace("{{MODEL_ANSWER}}", Model_Answer_fn)
    tn_prompt = PROMPT.replace("{{QUESTION}}", Question).replace("{{GROUND_TRUTH_ANSWER}}", Ground_Truth_Answer).replace("{{MODEL_ANSWER}}", Model_Answer_tn)

    fn_chat_response = client.chat.completions.create(
        model=VERIFIER_MODEL_NAME,
        messages=[
            {"role": "user", "content": fn_prompt},
        ],
        max_tokens=512 if "think" in VERIFIER_MODEL_NAME.lower() or "reasoning" in VERIFIER_MODEL_NAME.lower() else 8,
        temperature=0,
        top_p=1,
    )

    tn_chat_response = client.chat.completions.create(
        model=VERIFIER_MODEL_NAME,
        messages=[
            {"role": "user", "content": tn_prompt},
        ],
        max_tokens=512 if "think" in VERIFIER_MODEL_NAME.lower() or "reasoning" in VERIFIER_MODEL_NAME.lower() else 8,
        temperature=0,
        top_p=1,
    )

    tn_response_text = tn_chat_response.choices[0].message.content
    fn_response_text = fn_chat_response.choices[0].message.content

    fn_processed_response = fn_response_text
    tn_processed_response = tn_response_text
    
    if "think" in VERIFIER_MODEL_NAME.lower() or "reasoning" in VERIFIER_MODEL_NAME.lower():
        if "Answer:" in fn_response_text:
            fn_processed_response = fn_response_text.split("Answer:")[1].strip()
        if "Answer:" in tn_response_text:
            tn_processed_response = tn_response_text.split("Answer:")[1].strip()
    
    fn_is_correct = 'true' in fn_processed_response.lower()
    tn_is_correct = 'false' in tn_processed_response.lower()
    
    if fn_is_correct:
        fn_correct += 1
    else:
        print(f"Wrong FN at id: {entry['id']}: {entry['fn_output']} vs {entry['ground_truth']}")

    if tn_is_correct:
        tn_correct += 1
    else:
        print(f"Wrong TN at id: {entry['id']}: {entry['tn_output']} vs {entry['ground_truth']}")

    results.append({
        "id": entry["id"],
        "question": Question,
        "ground_truth": Ground_Truth_Answer,
        "fn_output": Model_Answer_fn,
        "tn_output": Model_Answer_tn,
        "fn_full_response": fn_response_text,
        "tn_full_response": tn_response_text,
        "fn_processed_response": fn_processed_response,
        "tn_processed_response": tn_processed_response,
        "fn_is_correct": fn_is_correct,
        "tn_is_correct": tn_is_correct
    })

output_filename = f"{VERIFIER_MODEL_NAME.replace('/', '_')}_v1_results.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Results saved to {output_filename}")
print(f"True Negative Correct: {tn_correct}/{len(benchmarks)}")
print(f"False Negative Correct: {fn_correct}/{len(benchmarks)}")