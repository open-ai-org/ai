# ai

GPU-accelerated ML CLI. Train, infer, quantize, and serve LLMs. Zero Python, one binary.

Powered by the [mongoose](https://github.com/open-ai-org/mongoose) GPU compute engine.

## Install

```bash
go install github.com/open-ai-org/ai@latest
```

Or build from source:

```bash
git clone https://github.com/open-ai-org/ai.git
cd ai
go build -o ai .
```

## Quick Start

```bash
# Download a model
ai pull Qwen/Qwen2.5-0.5B

# Chat
ai chat Qwen2.5-0.5B

# Generate text
ai infer Qwen2.5-0.5B "The meaning of life is"

# Start an OpenAI-compatible API server
ai serve Qwen2.5-0.5B
```

## Commands

### Training
| Command | Description |
|---------|-------------|
| `ai train data=<file>` | Train from scratch |
| `ai finetune model=<name> data=<file>` | Fine-tune pretrained model |
| `ai resume` | Continue training from checkpoint |

### Evaluation & Inference
| Command | Description |
|---------|-------------|
| `ai chat <model>` | Interactive chat |
| `ai infer <model> "prompt"` | Generate text |
| `ai eval model=<name> data=<file>` | Validation pass (loss + perplexity) |
| `ai benchmark <model>` | Inference throughput and latency |

### Optimization
| Command | Description |
|---------|-------------|
| `ai quantize <model> [q8\|q4\|f16]` | Reduce precision |
| `ai prune <model>` | Remove low-magnitude weights (50% default) |
| `ai prune <model> --sparsity 0.7 --structured` | Structured head pruning |
| `ai convert gguf <model>` | Export to GGUF (for Ollama) |
| `ai merge <base> <adapters>` | Merge LoRA into base model |

### Data
| Command | Description |
|---------|-------------|
| `ai dataset inspect <file>` | Dataset statistics and recommendations |
| `ai dataset split <file>` | Partition into train/val/test |
| `ai dataset augment <file>` | Dedup, lowercase, repeat, shuffle |

### Tuning & Search
| Command | Description |
|---------|-------------|
| `ai sweep data=<file> lr=1e-4,3e-4,6e-4` | Hyperparameter search |
| `ai distill teacher=<model> data=<file>` | Knowledge distillation |
| `ai profile` | Per-op GPU timing breakdown |

### Deployment
| Command | Description |
|---------|-------------|
| `ai serve <model>` | OpenAI-compatible API server |

### Introspection
| Command | Description |
|---------|-------------|
| `ai explain <model> "prompt"` | Token-level attribution |
| `ai checkpoint ls` | List training checkpoints |
| `ai checkpoint diff <a> <b>` | Compare checkpoints |
| `ai bench` | Raw GPU compute benchmark |
| `ai gpus` | Detect and calibrate hardware |

### Models
| Command | Description |
|---------|-------------|
| `ai pull <org/model>` | Download from HuggingFace |
| `ai models` | List downloaded models |
| `ai info <model>` | Show model architecture |

## Performance

### Training convergence — dim=512, RTX 5090

```
step 1     loss 6.17
step 100   loss 2.59   floor 2.37
step 300   loss 2.05   floor 1.76
step 500   loss 1.95   floor 1.29   365 steps/s
```

### Inference — Qwen2.5-0.5B (200 tokens)

| Platform | tok/s |
|----------|-------|
| RTX 5090 Q8 | 182 |
| M4 Max | 54 |

- Automatic quantization: Q8 for models <4B params, Q4 for 7B+
- Metal 4 `matmul2d` TensorOp on macOS 26+
- Fused dequant-matvec kernels — zero intermediate buffers
- Custom Metal/CUDA compute shaders for RMSNorm, RoPE, GQA attention, SiLU

## GPU Support

| Backend | Hardware | Build |
|---------|----------|-------|
| Metal | Apple Silicon (M1+) | `go build` on macOS |
| CUDA | NVIDIA GPUs | `go build` on Linux with CUDA toolkit |
| CPU | Any | `CGO_ENABLED=0 go build` |

## Dependencies

- [mongoose](https://github.com/open-ai-org/mongoose) — GPU compute engine
- [gguf](https://github.com/open-ai-org/gguf) — GGUF + SafeTensors I/O
- [tokenizer](https://github.com/open-ai-org/tokenizer) — BPE tokenizer

## License

MIT
