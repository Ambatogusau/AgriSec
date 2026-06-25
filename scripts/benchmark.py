"""
ADTC-style local benchmark: tokens/sec, peak RAM, latency, for self-testing
ahead of/alongside the official ADTC model profiler.

IMPORTANT: this version targets the Ollama-based src/inference.py. Pass an
Ollama model TAG, not a .gguf file path -- e.g.:

    ollama pull qwen2.5:1.5b        # one-time, downloads/registers the model
    python scripts/benchmark.py --model qwen2.5:1.5b --runs 5

Output: prints a summary and writes results to reports/benchmark_results.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference import LocalLLM  # noqa: E402

PROMPTS = [
    "What is the best time to plant maize in Northern Nigeria?",
    "Ina lokacin da ya dace a shuka masara a Arewacin Najeriya?",  # Hausa
    "My tomato leaves are turning yellow with brown spots. What could be wrong and what should I do?",
    "Recommend a fertilizer schedule for rice grown on loamy soil.",
    "How do I control fall armyworm in my maize farm without expensive pesticides?",
]

RAM_BUDGET_GB = 7.0


def get_peak_rss_mb(proc: psutil.Process):
    return proc.memory_info().rss / (1024 * 1024)


def run_benchmark(model_name: str, runs: int):
    proc = psutil.Process(os.getpid())
    print(f"Connecting to Ollama model: {model_name}")
    load_start = time.time()
    llm = LocalLLM(model_name)

    # Fire one short throwaway request first -- with Ollama this is what
    # actually triggers the model load into memory; without this, your
    # "load time" measurement is meaningless (LocalLLM.__init__ itself does
    # no network call) and your FIRST timed run silently eats the load cost.
    warmup = llm.generate("Say OK.", max_tokens=5)
    load_time = time.time() - load_start
    if warmup["tokens"] == 0 and "error" in warmup["text"].lower():
        print(f"WARNING: warmup call failed -- {warmup['text']}")
        print("Check that `ollama serve` is running and the model has been pulled:")
        print(f"  ollama pull {model_name}")
        sys.exit(1)
    print(f"Model responded in {load_time:.1f}s (includes first-load cost)")

    results = []
    peak_rss_mb = get_peak_rss_mb(proc)

    for i in range(runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        gen = llm.generate(prompt, max_tokens=200)
        peak_rss_mb = max(peak_rss_mb, get_peak_rss_mb(proc))
        results.append(gen)
        print(f"  run {i+1}/{runs}: {gen['tokens']} tok in {gen['elapsed_sec']:.2f}s "
              f"-> {gen['tokens_per_sec']:.2f} tok/s | peak RSS so far: {peak_rss_mb:.0f} MB")

    tps_values = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]
    if not tps_values:
        print("\nAll runs returned 0 tokens -- something is wrong with the Ollama connection "
              "or model name. No summary written.")
        sys.exit(1)

    summary = {
        "model_name": model_name,
        "runs": runs,
        "first_response_time_sec": round(load_time, 2),
        "tps_mean": round(statistics.mean(tps_values), 2),
        "tps_min": round(min(tps_values), 2),
        "tps_max": round(max(tps_values), 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "peak_rss_gb": round(peak_rss_mb / 1024, 3),
        "ram_budget_gb": RAM_BUDGET_GB,
        "seff_estimate": round(100 * ((RAM_BUDGET_GB - peak_rss_mb / 1024) / RAM_BUDGET_GB), 1),
        "sperf_estimate_at_tps_ref_15": round(100 * (statistics.mean(tps_values) / 15.0), 1),
        "note": "peak_rss_mb is the RAM of THIS PYTHON PROCESS only -- it does NOT include "
                "the separate `ollama serve` process, which holds the actual model weights. "
                "For an accurate Seff number, also check `ollama ps` and add that process's "
                "RAM, or use the official ADTC profiler which should measure system-wide. "
                "Thermal/throttle and CPU package temp are also NOT measured here -- "
                "use `sensors` (lm-sensors) or the official ADTC profiler for that.",
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen2.5:1.5b")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    summary = run_benchmark(args.model, args.runs)

    print("\n=== Benchmark Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
