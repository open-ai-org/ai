# CLAUDE.md — ai

## What This Is

Command-line interface for GPU-accelerated ML. Train, infer, quantize, serve. Powered by the mongoose engine ecosystem. Zero Python, one binary.

## Build

```bash
go build -o ai .
```

On Linux with CUDA:
```bash
CGO_CFLAGS="-I/usr/local/cuda/include" CGO_LDFLAGS="-L/usr/local/cuda/lib64" go build -o ai .
```

CUDA kernels (optional, enables fused Q8/Q4 inference):
```bash
cd ../mongoose/kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90
cp libmongoose_kernels.so ../../ai/
```

Metal 4 kernels (macOS, optional — enables Metal 4 matmul2d TensorOp):
```bash
cd ../mongoose/kernels
xcrun metal -std=metal4.0 -O2 -c gemm_metal4.metal -o gemm_metal4.air
xcrun metallib -o gemm_metal4.metallib gemm_metal4.air
cp gemm_metal4.metallib ../../ai/
```

## Usage

```bash
ai pull Qwen/Qwen2.5-0.5B          # download from HuggingFace
ai infer Qwen2.5-0.5B "Hello"      # generate text
ai serve Qwen2.5-0.5B              # OpenAI-compatible API
ai quantize Qwen2.5-0.5B q8        # quantize to INT8 GGUF
ai convert gguf Qwen2.5-0.5B       # convert to GGUF for Ollama
ai train data=corpus.txt            # train from scratch
ai bench                            # raw GPU benchmark
ai gpus                             # detect and calibrate GPUs
```

## Architecture

- `main.go` — command dispatch, usage
- `infer_gpu.go` — GPU inference with tiered dispatch:
  - Metal fused compute (custom kernels + Metal 4 matmul2d): Q8 or Q4
  - CUDA fused Q8/Q4 matvec: custom kernels + cuBLAS
  - GPU-resident tier 2: cuBLAS/MPS matmul + CPU attention
  - CPU streaming tier 3: pure Go fallback
- `infer.go` — CPU-only text generation
- `serve.go` — OpenAI-compatible API server (chat, completions, embeddings)
- `train_unified.go` — unified training entry point
- `train_cuda.go` — CUDA training with helix optimizer
- `train_finetune.go` — fine-tuning pretrained models
- `commands.go` — pull, models, info, bench, gpus, convert, export
- `quantize.go` — quantize safetensors to GGUF (Q8_0, Q4_0, F16, F32)
- `globals.go` — shared config, engine selection (`selectEngine`)
- `autodetect.go` — hardware detection and model profiling
- `checkpoint.go` — save/resume training state
- `merge.go` — merge LoRA adapters into base model
- `benchmark.go` — model inference profiling
- `dataset.go` — dataset loading and inspection
- `eval.go` — validation (loss + perplexity)

## Inference Tier Selection

The `cmdInferGPU` function selects the fastest available path:

1. **Metal fused compute** (macOS): Custom Metal compute shaders (RMSNorm, RoPE, GQA attention, SiLU) + Metal 4 `matmul2d` TensorOp or fused Q8/Q4 matvec. One command buffer per token. Weights quantized to Q8 (<4B params) or Q4 (>4B params) on load.
2. **CUDA fused Q8/Q4** (Linux): Custom CUDA kernels for RMSNorm, RoPE, decode attention, SiLU, fused Q8/Q4 dequant-matvec. Same >4B param gate for Q4.
3. **GPU-resident tier 2**: Weights in VRAM, matmuls via cuBLAS/MPS, attention on CPU.
4. **CPU streaming tier 3**: Weights loaded from disk per layer, pure Go.

## Dependencies

- `github.com/open-ai-org/mongoose` — GPU compute engine
- `github.com/open-ai-org/gguf` — model serialization (GGUF + SafeTensors)
- `github.com/open-ai-org/tokenizer` — BPE tokenizer
- `github.com/open-ai-org/helix` — DNA optimizer (optional, --helix flag)
- `github.com/open-ai-org/needle` — INT8 kernels (optional, --needle flag)

## Test

```bash
go test -v ./...
```
