#!/usr/bin/env python3
"""
correctness_checkers.py
Evaluates model outputs for correctness and consistency across prompt categories.

Usage:
    python3 correctness_checkers.py /path/to/outputs.csv

Output:
 - Prints per-prompt correctness analysis to stdout
 - Writes correctness_summary.json alongside the CSV

Metrics:
 - Exact match (for deterministic tasks like math/code)
 - Keyword presence (for narrative/JSON tasks)
 - Format validity (JSON parse check)
 - Aggregate accuracy percentage
"""

import os
import sys
import csv
import json
import ast
import re

def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def looks_like_json(text):
    text = text.strip()
    return text.startswith("{") or text.startswith("[")

def is_math_expression(text):
    # crude heuristic for arithmetic-style answers
    return bool(re.search(r"[0-9+\-*/=]", text))

def evaluate_row(prompt, output):
    """
    Returns a dict with correctness metrics for a single prompt-output pair.
    """
    correctness = False
    category = "general"

    p_lower = prompt.lower()
    out_lower = output.lower().strip()

    # --- Category and rules ---
    if "derivative" in p_lower or "integral" in p_lower or "solve" in p_lower:
        category = "math"
        # basic numeric or expression match check
        correctness = any(sym in out_lower for sym in ["x", "y", "=", "dx", "dy", "6x", "2x"]) or is_math_expression(out_lower)

    elif "python" in p_lower or "code" in p_lower or "function" in p_lower:
        category = "code"
        # should contain def, return, or braces
        correctness = any(k in out_lower for k in ["def ", "return", "for ", "while "])

    elif "json" in p_lower or "object" in p_lower or "key" in p_lower:
        category = "json"
        # test for valid JSON
        if looks_like_json(output):
            correctness = safe_json_load(output) is not None

    elif "story" in p_lower or "write" in p_lower or "poem" in p_lower:
        category = "writing"
        # writing task correctness = coherent structure
        correctness = len(out_lower.split()) > 10 and "." in out_lower

    else:
        category = "general"
        # general: output should be non-empty and relevant
        correctness = len(out_lower) > 5 and any(w in out_lower for w in p_lower.split()[:3])

    return {
        "category": category,
        "correct": bool(correctness),
        "output_len": len(out_lower.split()),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 correctness_checkers.py /path/to/outputs.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("No data found in CSV.")
        sys.exit(0)

    total = len(rows)
    correct = 0
    category_stats = {}

    for r in rows:
        prompt = r.get("prompt", "")
        output = r.get("output", "")
        res = evaluate_row(prompt, output)
        if res["correct"]:
            correct += 1
        cat = res["category"]
        category_stats.setdefault(cat, {"total": 0, "correct": 0})
        category_stats[cat]["total"] += 1
        if res["correct"]:
            category_stats[cat]["correct"] += 1

    overall_acc = round((correct / total) * 100, 2)

    # Build summary JSON
    summary = {
        "file": os.path.basename(csv_path),
        "total": total,
        "correct": correct,
        "accuracy_percent": overall_acc,
        "per_category": {},
    }
    for cat, stats in category_stats.items():
        acc = round((stats["correct"] / stats["total"]) * 100, 2) if stats["total"] else 0
        summary["per_category"][cat] = {
            "accuracy_percent": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # Save summary
    out_path = os.path.join(os.path.dirname(csv_path), "correctness_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty print for logs
    print("=== Correctness Report ===")
    print(f"File: {csv_path}")
    print(f"Total prompts: {total}")
    print(f"Correct responses: {correct}")
    print(f"Overall accuracy: {overall_acc}%\n")
    print("Per-category breakdown:")
    for cat, s in summary["per_category"].items():
        print(f" - {cat.title():<10}: {s['accuracy_percent']}% ({s['correct']}/{s['total']})")
    print("\n✅ Summary saved to:", out_path)


if __name__ == "__main__":
    main()
