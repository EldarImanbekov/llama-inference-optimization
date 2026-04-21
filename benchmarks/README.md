# LLaMA Inference Optimization

Progressive optimization of LLaMA 1B inference — from baseline measurement to kernel-level improvements.

## Goal

Identify and eliminate performance bottlenecks in LLaMA 1B inference using profiling-driven optimization. Same methodology as the [CUDA GEMM project](https://github.com/EldarImanbekov/cuda-gemm) — measure first, optimize second.

## Structure

```
benchmarks/     — timing and throughput measurement
kernels/        — custom CUDA kernels
profiling/      — Nsight profiling scripts  
results/        — benchmark outputs (JSON)
```

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 — Baseline | Run vanilla LLaMA, measure tok/s and latency | In progress |
| 2 — Profiling | Nsight profile to find bottlenecks | Pending |
| 3 — Optimization | Apply targeted kernel optimizations | Pending |
| 4 — Results | Before/after comparison | Pending |

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline benchmark
python benchmarks/baseline.py
```

## Baseline Results

*Coming soon — running on Tesla T4*

## Hardware

- GPU: Tesla T4 (Google Colab)
- Target: NVIDIA A100 (Azure)
