import os
import json
import time
import argparse
import csv
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from correctness_checkers import evaluate_row   # Pi version


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompts", required=True)
parser.add_argument("--offload_dir", required=True)
parser.add_argument("--results_root", required=True)
parser.add_argument("--max_new_tokens", type=int, default=12)
args = parser.parse_args()


# Validate paths
if not os.path.exists(args.prompts):
    raise FileNotFoundError(f"Prompts file not found: {args.prompts}")

os.makedirs(args.offload_dir, exist_ok=True)
os.makedirs(args.results_root, exist_ok=True)


# Load prompts
with open(args.prompts, "r") as f:
    data = json.load(f)

if isinstance(data, dict):
    prompts = data.get("prompts", [])
else:
    prompts = data

if not prompts:
    raise ValueError("No prompts found in prompts file.")


# Output directory
run_name = (
    args.model.replace("/", "_") +
    "_run_" +
    os.path.basename(args.prompts).replace(".json", "") +
    "_" +
    time.strftime("%Y%m%d_%H%M%S")
)

run_dir = os.path.join(args.results_root, run_name)
os.makedirs(run_dir, exist_ok=True)

log_path = os.path.join(run_dir, "run.log")
telemetry_path = os.path.join(run_dir, "telemetry.json")
outputs_csv_path = os.path.join(run_dir, "outputs.csv")
summary_json_path = os.path.join(run_dir, "summary.json")
correctness_summary_path = os.path.join(run_dir, "correctness_summary.json")


# Logging helper
def log(msg):
    print(msg)
    with open(log_path, "a") as lf:
        lf.write(msg + "\n")


log("BigBoss Baseline Runner (Raspberry Pi)")
log(f"Model: {args.model}")
log(f"Prompts: {args.prompts}")
log(f"Output folder: {run_dir}")


# Telemetry collector (CPU, RAM, SWAP)
def collect_telemetry():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "swap_percent": psutil.swap_memory().percent,
        "timestamp": time.time()
    }

telemetry_history = []


# Load tokenizer/model
log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model)

log("Loading model (CPU offload-safe)...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map="auto",
    offload_folder=args.offload_dir,
    torch_dtype="auto"
)

log("Model loaded.")


# Run benchmark
log("Starting evaluation...")

with open(outputs_csv_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "output", "latency_s", "correct"])

    correct_total = 0

    for i, prompt in enumerate(prompts):
        log(f" [{i+1}/{len(prompts)}] Running prompt...")
        telemetry_history.append(collect_telemetry())

        encoded = tokenizer(prompt, return_tensors="pt")

        start = time.time()
        out_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
        latency = round(time.time() - start, 3)

        out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        row_eval = evaluate_row(prompt, out_text)
        is_correct = row_eval["correct"]
        if is_correct:
            correct_total += 1

        writer.writerow([prompt, out_text, latency, is_correct])


# Write summary.json
summary = {
    "model": args.model,
    "prompts_file": args.prompts,
    "timestamp": time.time(),
    "total_prompts": len(prompts),
    "correct": correct_total,
    "accuracy_percent": round(correct_total / len(prompts) * 100, 2)
}

with open(summary_json_path, "w") as f:
    json.dump(summary, f, indent=2)


# Write telemetry.json
with open(telemetry_path, "w") as f:
    json.dump(telemetry_history, f, indent=2)


# Generate correctness_summary.json
with open(outputs_csv_path, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

cats = {}
for r in rows:
    prompt = r["prompt"]
    output = r["output"]
    res = evaluate_row(prompt, output)
    cat = res["category"]

    cats.setdefault(cat, {"total": 0, "correct": 0})
    cats[cat]["total"] += 1
    if res["correct"]:
        cats[cat]["correct"] += 1

cat_summary = {
    "overall_accuracy": summary["accuracy_percent"],
    "per_category": {
        k: {
            "accuracy_percent": round(v["correct"] / v["total"] * 100, 2),
            "correct": v["correct"],
            "total": v["total"]
        }
        for k, v in cats.items()
    }
}

with open(correctness_summary_path, "w") as f:
    json.dump(cat_summary, f, indent=2)


# ZIP the whole folder
import shutil

zip_path = run_dir + ".zip"
shutil.make_archive(run_dir, "zip", run_dir)

log(f" Results packaged into: {zip_path}")
log("Benchmark completed!")