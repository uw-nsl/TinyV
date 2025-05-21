import concurrent.futures as cf
import concurrent
import json, os, re, sys, threading, time, warnings
from collections import deque
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
import argparse

# parameters
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Please fill in your own API keys
XAI_API_KEYS = [
    ""
]

BASE_URL, MODEL_NAME, REASONING_EFFORT = (
    "https://api.x.ai/v1",
    "grok-3-mini-beta",
    "high",
)
MAX_WORKERS = 2000
MAX_RETRIES, MAX_ATTEMPTS, BASE_DELAY_S = 8, 4, 1  # MAX_RETRIES for API errors only
REQUEST_SOFT_TIMEOUT = 120
PROMPT_DIR = "./prompts"

# globals
env = Environment(loader=FileSystemLoader(PROMPT_DIR))
key_cycle, key_lock = cycle(XAI_API_KEYS), threading.Lock()

def get_args():
    parser = argparse.ArgumentParser(description="Unified Tagging Manager with Grok API.")
    parser.add_argument("--tag_mission", type=str, default="classification", 
                       help="The tagging mission.", 
                       choices=["classification"])
    parser.add_argument("--input_file", type=str, required=True, help="Input dataset file name")
    parser.add_argument("--checkpoint_every", type=int, default=2000, help="Save checkpoint every n batches")
    parser.add_argument("--debug", action="store_true", help="Debug mode. Only process the first 1000 samples.")
    parser.add_argument("--save_as", type=str, default="json", choices=["json", "jsonl"], 
                       help="Save the generated responses as a what kind of file")
    return parser.parse_args()

def template_generator(input, mission):
    # Use classification.md as the template
    if mission == "classification":
        template = env.get_template('classification.md').render(
            GROUND_TRUTH_ANSWER=input['ground_truth'],
            MODEL_ANSWER=input['extracted_model_output']
        )
        return template
    else:
        raise ValueError("Invalid mission. Available missions: classification")

def process_engine_responses(response, item, mission):
    # Only process classification tasks
    if mission == "classification":
        # Only extract primary and secondary categories
        # <primary_category>...</primary_category>
        # <second_category>...</second_category>
        try:
            primary_match = re.search(r"<primary_category>\s*(.*?)\s*</primary_category>", response, re.DOTALL)
            second_match = re.search(r"<second_category>\s*(.*?)\s*</second_category>", response, re.DOTALL)
            item['grok_primary_category'] = primary_match.group(1).strip() if primary_match else None
            # If second_category is empty, "None", or an empty string, set to None
            if second_match:
                val = second_match.group(1).strip()
                if val and val.lower() != "none":
                    item['grok_second_category'] = val
                else:
                    item['grok_second_category'] = None
            else:
                item['grok_second_category'] = None
        except Exception as e:
            print(f"[classification_grok.py] Failed to process item with error: {str(e)}")
            print(f"[classification_grok.py] Raw response from LLM tagger: {response}")
            item['grok_primary_category'] = None
            item['grok_second_category'] = None
    return item

def handle_one(item: Dict, mission: str) -> Union[Dict[str, str], str, None]:
    delay = BASE_DELAY_S
    for retry in range(1, MAX_RETRIES + 1):
        try:
            with key_lock:
                api_key = next(key_cycle)
            client = OpenAI(api_key=api_key, base_url=BASE_URL)

            with cf.ThreadPoolExecutor(max_workers=1) as exec:
                fut = exec.submit(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    reasoning_effort=REASONING_EFFORT,
                    messages=[
                        {"role": "user", "content": template_generator(item, mission)}
                    ],
                    temperature=0.5,
                )
                resp = fut.result(timeout=REQUEST_SOFT_TIMEOUT)
            
            return process_engine_responses(resp.choices[0].message.content, item, mission)
            
        except concurrent.futures.TimeoutError:
            tqdm.write(f"[ID {item.get('index', 'unknown')}] API soft timeout (retry {retry}/{MAX_RETRIES})")
            if retry == MAX_RETRIES:
                # If all retries are used, return a dictionary with None values instead of discarding
                item['grok_primary_category'] = None
                item['grok_second_category'] = None
                return item
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            tqdm.write(
                f"[ID {item.get('id', 'unknown')}] API error: {e} (retry {retry}/{MAX_RETRIES})"
            )
            if retry == MAX_RETRIES:
                # If all retries are used, return a dictionary with None values instead of discarding
                item['grok_primary_category'] = None
                item['grok_second_category'] = None
                return item
            time.sleep(delay)
            delay *= 2
    return None

def append_jsonl(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    args = get_args()
    print(f"[classification_grok.py] Unified Tagging Manager with Grok API. Arguments: {args}")

    inp, out = Path(args.input_file), Path(args.input_file[:args.input_file.rfind('.')] + f"_{args.tag_mission}.{args.save_as}")
    checkpoint_file = Path(args.input_file[:args.input_file.rfind('.')] + f"_{args.tag_mission}_checkpoint.json")

    # Load dataset
    if not args.debug:
        items = json.loads(inp.read_text())
    else:
        warnings.warn("Debug mode enabled. Only processing the first 1000 samples.")
        items = json.loads(inp.read_text())[:1000]

    # Load checkpoint if exists
    if checkpoint_file.exists():
        done_items = json.loads(checkpoint_file.read_text())
        done_ids = {item.get('index', str(i)) for i, item in enumerate(done_items)}
        pend = deque([obj for obj in items if obj.get('index', str(i)) not in done_ids])
        print(f"[classification_grok.py] Checkpoint file found. Resuming from last checkpoint with {len(done_items)} items processed.")
    else:
        pend = deque(items)
        done_items = []

    total = len(pend)
    overall = tqdm(total=total, desc="Overall", ncols=80)

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        running = {}
        while pend and len(running) < MAX_WORKERS:
            it = pend.popleft()
            fut = pool.submit(handle_one, it, args.tag_mission)
            running[fut] = it

        while running:
            done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                item = running.pop(fut)
                try:
                    result = fut.result()
                    if isinstance(result, dict):
                        done_items.append(result)
                        if len(done_items) % args.checkpoint_every == 0:
                            checkpoint_file.write_text(json.dumps(done_items, ensure_ascii=False, indent=2))
                            print(f"[classification_grok.py] Checkpoint saved after processing {len(done_items)} items")
                        overall.update(1)
                    else:
                        tqdm.write(f"[ID {item.get('index', 'unknown')}] failed, retrying")
                        pend.append(item)
                except Exception as e:
                    tqdm.write(f"[ID {item.get('index', 'unknown')}] fatal error: {e}")
                    pend.append(item)

                if pend:
                    nxt = pend.popleft()
                    fut2 = pool.submit(handle_one, nxt, args.tag_mission)
                    running[fut2] = nxt

    overall.close()
    
    # Sort done_items by index before saving
    done_items.sort(key=lambda x: int(x.get('index', 0)))
    
    # Save final results
    if args.save_as == "json":
        out.write_text(json.dumps(done_items, ensure_ascii=False, indent=2))
    else:
        append_jsonl(out, done_items)
    
    # Remove checkpoint file
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("[classification_grok.py] Final dataset saved. Checkpoint removed.")
    
    print("\nAll done.")

if __name__ == "__main__":
    main()
