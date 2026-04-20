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

# Generate text
ai infer Qwen2.5-0.5B "The meaning of life is"

# Start an OpenAI-compatible API server
ai serve Qwen2.5-0.5B
```

## Commands

| Command | Description |
|---------|-------------|
| `ai infer <model> "prompt"` | Generate text |
| `ai serve <model>` | OpenAI-compatible API server |
| `ai pull <org/model>` | Download from HuggingFace |
| `ai models` | List downloaded models |
| `ai info <model>` | Show model architecture |
| `ai train data=<file>` | Train from scratch |
| `ai train model=<name> data=<file>` | Fine-tune pretrained model |
| `ai quantize <model> [q8\|q4\|f16]` | Quantize model weights |
| `ai convert gguf <model>` | Convert to GGUF (for Ollama) |
| `ai merge <base> <adapters>` | Merge LoRA into base model |
| `ai bench` | Raw GPU benchmark |
| `ai gpus` | Detect and calibrate GPUs |

## Performance

Inference on Qwen2.5-0.5B:

| Platform | tok/s |
|----------|-------|
| PyTorch MPS (M1 Pro) | 3.3 |
| ai Metal Q8 (M1 Pro) | 57 |
| ai CUDA Q8 (RTX 5070 Ti) | 99 |

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
