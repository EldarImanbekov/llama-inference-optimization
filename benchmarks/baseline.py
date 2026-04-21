"""
baseline.py

Establishes the performance baseline for LLaMA 1B inference.
Run this BEFORE any optimizations to get your starting numbers.

Measures:
- Tokens per second (throughput)
- Time to first token (latency)
- Memory usage (peak GPU memory)
- Per-token latency (ms/token)

Usage:
    python baseline.py
"""

import torch
import time
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODEL_ID    = "meta-llama/Llama-3.2-1B"   # 1B parameter model
MAX_TOKENS  = 200                           # tokens to generate per run
NUM_RUNS    = 5                             # runs to average over
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Test prompts — varied length and complexity
PROMPTS = [
    "Explain what a GPU is in simple terms.",
    "What is the difference between CPU and GPU computing?",
    "How does matrix multiplication work?",
    "Describe the attention mechanism in transformer models.",
    "What is memory bandwidth and why does it matter for deep learning?",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_gpu_memory_mb():
    """Returns current peak GPU memory usage in MB."""
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_gpu_memory():
    """Resets peak memory tracking."""
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def warmup(model, tokenizer, device):
    """
    Warmup run — GPU needs to warm up for accurate timing.
    Same reason we did warmup in the GEMM benchmark.
    """
    print("Warming up...")
    inputs = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)
    print("Warmup done.\n")


def measure_inference(model, tokenizer, prompt, max_new_tokens, device):
    """
    Measures a single inference run.
    Returns dict with timing and memory stats.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    reset_gpu_memory()

    # Sync GPU before timing — same principle as cudaDeviceSynchronize()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy decoding — deterministic
            temperature=1.0,
            use_cache=True,         # KV cache enabled (default)
        )

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time_s   = end - start
    output_len     = outputs.shape[1] - input_len
    tokens_per_sec = output_len / total_time_s
    ms_per_token   = (total_time_s / output_len) * 1000
    peak_memory_mb = get_gpu_memory_mb()

    return {
        "prompt_tokens":   input_len,
        "generated_tokens": output_len,
        "total_time_s":    round(total_time_s, 4),
        "tokens_per_sec":  round(tokens_per_sec, 2),
        "ms_per_token":    round(ms_per_token, 3),
        "peak_memory_mb":  round(peak_memory_mb, 1),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LLaMA Inference Baseline Benchmark")
    print("=" * 60)
    print(f"Model:    {MODEL_ID}")
    print(f"Device:   {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Runs:     {NUM_RUNS} per prompt")
    print(f"Tokens:   {MAX_TOKENS} generated per run")
    print("=" * 60)
    print()

    # ── Load model ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading model (this takes a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # FP16 — standard for inference
        device_map="auto",           # automatically places on GPU
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")

    # ── Warmup ──
    warmup(model, tokenizer, DEVICE)

    # ── Benchmark each prompt ──
    all_results = []

    for i, prompt in enumerate(PROMPTS):
        print(f"Prompt {i+1}/{len(PROMPTS)}: \"{prompt[:50]}...\"")
        run_results = []

        for run in range(NUM_RUNS):
            result = measure_inference(model, tokenizer, prompt, MAX_TOKENS, DEVICE)
            run_results.append(result)
            print(f"  Run {run+1}: {result['tokens_per_sec']:6.1f} tok/s  "
                  f"{result['ms_per_token']:6.2f} ms/tok  "
                  f"{result['peak_memory_mb']:6.0f} MB")

        # Average across runs
        avg = {
            "prompt":           prompt[:50],
            "prompt_tokens":    run_results[0]["prompt_tokens"],
            "generated_tokens": run_results[0]["generated_tokens"],
            "avg_tokens_per_sec": round(sum(r["tokens_per_sec"] for r in run_results) / NUM_RUNS, 2),
            "avg_ms_per_token":   round(sum(r["ms_per_token"]   for r in run_results) / NUM_RUNS, 3),
            "avg_memory_mb":      round(sum(r["peak_memory_mb"] for r in run_results) / NUM_RUNS, 1),
        }
        all_results.append(avg)
        print(f"  AVG:  {avg['avg_tokens_per_sec']:6.1f} tok/s  "
              f"{avg['avg_ms_per_token']:6.2f} ms/tok\n")

    # ── Summary ──
    print("=" * 60)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Prompt':<35} {'Tok/s':>8} {'ms/tok':>8} {'Mem MB':>8}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        print(f"{r['prompt']:<35} {r['avg_tokens_per_sec']:>8.1f} "
              f"{r['avg_ms_per_token']:>8.2f} {r['avg_memory_mb']:>8.0f}")

    overall_tps = sum(r["avg_tokens_per_sec"] for r in all_results) / len(all_results)
    overall_mpt = sum(r["avg_ms_per_token"]   for r in all_results) / len(all_results)
    overall_mem = sum(r["avg_memory_mb"]       for r in all_results) / len(all_results)

    print(f"\n{'OVERALL AVERAGE':<35} {overall_tps:>8.1f} "
          f"{overall_mpt:>8.2f} {overall_mem:>8.0f}")
    print("=" * 60)

    # ── Save results to JSON ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp":     timestamp,
        "model":         MODEL_ID,
        "device":        DEVICE,
        "gpu_name":      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "max_tokens":    MAX_TOKENS,
        "num_runs":      NUM_RUNS,
        "results":       all_results,
        "overall": {
            "avg_tokens_per_sec": round(overall_tps, 2),
            "avg_ms_per_token":   round(overall_mpt, 3),
            "avg_memory_mb":      round(overall_mem, 1),
        }
    }

    os.makedirs("results", exist_ok=True)
    fname = f"results/baseline_{timestamp}.json"
    with open(fname, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {fname}")
    print("This is your baseline. Every optimization gets measured against these numbers.")


if __name__ == "__main__":
    main()