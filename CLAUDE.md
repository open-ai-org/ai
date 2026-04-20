# CLAUDE.md — ai

## What This Is

Command-line interface for GPU-accelerated ML. Train, infer, quantize, serve. Powered by the mongoose engine ecosystem. Zero Python.

## Build

```bash
go build -o ai .
```

## Usage

```bash
ai pull Qwen/Qwen2.5-14B       # download from HuggingFace
ai bench                       # benchmark your GPU
ai infer qwen2-0.5b "Hello"    # generate text
ai quantize model q8           # quantize to INT8
ai serve model                 # OpenAI-compatible API server
ai convert gguf model          # convert to GGUF for Ollama
```

## Architecture

- `main.go` — command dispatch
- `commands.go` — pull, models, info, bench, gpus, convert, export
- `infer.go` — text generation (CPU streaming path, GPU graph dispatch planned)
- `train.go` — training commands (stubs — wiring to GraphTrainEngine in progress)
- `serve.go` — OpenAI-compatible API server (chat, completions, embeddings)
- `benchmark.go` — model inference profiling
- `globals.go` — shared config, engine selection
- `autodetect.go` — hardware detection and model profiling
- `checkpoint.go` — save/resume training state
- `merge.go` — merge LoRA adapters into base model
- `quantize.go` — quantize weights (F32 → Q8_0, Q4_0, F16)

## Dependencies

- `github.com/open-ai-org/mongoose` — GPU compute engine
- `github.com/open-ai-org/gguf` — model serialization
- `github.com/open-ai-org/tokenizer` — BPE tokenizer
- `github.com/open-ai-org/helix` — DNA optimizer (optional, --helix flag)
- `github.com/open-ai-org/needle` — INT8 kernels (optional, --needle flag)

## Known State

Training commands are stubs — the fused graph dispatch (GraphTrainEngine) needs to be wired in. Inference works on CPU streaming path. GPU inference via GraphTrainEngine is the next step.
