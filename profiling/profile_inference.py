"""
profile_inference.py

Profiles LLaMA 1B inference to find exactly where time is spent.
Uses PyTorch's built-in profiler — no Nsight needed for this step.

What this tells you:
- Which operations consume the most GPU time
- Whether you are compute-bound or memory-bound
- How much time attention vs FFN takes
- Whether KV cache reads are a bottleneck

Usage:
    python profiling/profile_inference.py

Output:
    - Console table sorted by GPU time
    - results/profile_<timestamp>.json
    - results/profile_trace_<timestamp>.json  (open in Chrome trace viewer)
"""

import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODEL_ID      = "meta-llama/Llama-3.2-1B"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PROFILE_STEPS = 50    # tokens to profile — enough to see patterns, not too slow
WARMUP_STEPS  = 10    # warmup tokens before profiling starts

# Single focused prompt for profiling
PROMPT = "Explain how matrix multiplication works in neural networks."


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LLaMA Inference Profiler")
    print("=" * 60)
    print(f"Model:   {MODEL_ID}")
    print(f"Device:  {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Profiling {PROFILE_STEPS} tokens after {WARMUP_STEPS} warmup")
    print("=" * 60)
    print()

    # ── Load model ──
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    # ── Tokenize ──
    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    # ── Warmup — run without profiling ──
    print(f"Warming up ({WARMUP_STEPS} tokens)...")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=WARMUP_STEPS,
            do_sample=False,
            use_cache=True,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    print("Warmup done.\n")

    # ── Profile ──
    print(f"Profiling {PROFILE_STEPS} tokens...")
    print("This takes a minute — profiler adds overhead...\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = f"results/profile_trace_{timestamp}.json"
    os.makedirs("results", exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=PROFILE_STEPS,
                do_sample=False,
                use_cache=True,
            )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ── Print top operations by CUDA time ──
    print("\n" + "=" * 60)
    print("  TOP OPERATIONS BY GPU TIME")
    print("=" * 60)
    print(f"{'Operation':<45} {'CUDA %':>7} {'CUDA ms':>9} {'Calls':>7}")
    print(f"{'-'*45} {'-'*7} {'-'*9} {'-'*7}")

    # Get key averages sorted by CUDA time
    key_avgs = prof.key_averages()
    total_cuda = sum(k.device_time for k in key_avgs)

    sorted_ops = sorted(
        [k for k in key_avgs if k.device_time > 0],
        key=lambda x: x.device_time,
        reverse=True
    )[:20]

    results_data = []
    for op in sorted_ops:
        pct = (op.device_time / total_cuda * 100) if total_cuda > 0 else 0
        ms  = op.device_time / 1000
        print(f"{op.key[:45]:<45} {pct:>6.1f}% {ms:>9.3f} {op.count:>7}")
        results_data.append({
            "operation": op.key,
            "cuda_pct":  round(pct, 2),
            "cuda_ms":   round(ms, 3),
            "calls":     op.count,
        })

    print("=" * 60)

    # ── Categorize into attention vs FFN vs other ──
    attn_ops  = ["aten::mm", "aten::bmm", "aten::matmul", "aten::scaled_dot_product_attention"]
    ffn_ops   = ["aten::mm", "aten::addmm"]

    attn_time = sum(k.device_time for k in key_avgs
                    if any(a in k.key for a in ["bmm","scaled_dot","softmax"]))
    mm_time   = sum(k.device_time for k in key_avgs
                    if any(a in k.key for a in ["aten::mm","aten::addmm"]))
    other_time = total_cuda - attn_time - mm_time

    print("\n  BREAKDOWN BY CATEGORY")
    print("=" * 60)
    if total_cuda > 0:
        print(f"  Matrix multiplies (mm/addmm): {mm_time/total_cuda*100:5.1f}%  — your GEMM kernels")
        print(f"  Attention ops (bmm/softmax):  {attn_time/total_cuda*100:5.1f}%  — KV cache + attention")
        print(f"  Other (norm, activations):    {other_time/total_cuda*100:5.1f}%  — layer norm, SiLU etc")
    print("=" * 60)

    # ── Memory bandwidth estimate ──
    print("\n  MEMORY ANALYSIS")
    print("=" * 60)
    total_mem_ops = sum(k.self_device_memory_usage for k in key_avgs if k.self_device_memory_usage > 0)
    print(f"  Peak GPU memory:    {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print(f"  Total CUDA time:    {total_cuda/1000:.1f} ms")
    print(f"  Avg per token:      {total_cuda/1000/PROFILE_STEPS:.2f} ms/token")
    print("=" * 60)

    # ── Save Chrome trace (open at chrome://tracing) ──
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace saved: {trace_file}")
    print("Open chrome://tracing in your browser and load this file")
    print("to see a visual timeline of every GPU operation.\n")

    # ── Save JSON summary ──
    summary = {
        "timestamp":       timestamp,
        "model":           MODEL_ID,
        "gpu":             torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "profile_tokens":  PROFILE_STEPS,
        "total_cuda_ms":   round(total_cuda / 1000, 3),
        "avg_ms_per_token": round(total_cuda / 1000 / PROFILE_STEPS, 3),
        "top_ops":         results_data,
        "breakdown": {
            "matmul_pct":  round(mm_time/total_cuda*100, 1) if total_cuda > 0 else 0,
            "attention_pct": round(attn_time/total_cuda*100, 1) if total_cuda > 0 else 0,
            "other_pct":   round(other_time/total_cuda*100, 1) if total_cuda > 0 else 0,
        }
    }

    summary_file = f"results/profile_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved: {summary_file}")
    print("\nThese numbers tell you WHERE to optimize.")
    print("The operation with the highest CUDA % is your target.")


if __name__ == "__main__":
    main()
