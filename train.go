package main

import (
	"fmt"
	"os"
)

func cmdTrain() {
	fmt.Println("tesseract train — train a model from scratch")
	fmt.Println()
	fmt.Println("Uses GraphTrainEngine: one fused GPU dispatch for forward + backward + optimizer.")
	fmt.Println("This is the path that beat PyTorch 2-280%. No helix, no needle — pure graph dispatch.")
	fmt.Println()
	fmt.Println("Usage: tesseract train --data <path> [flags]")
	fmt.Println()
	fmt.Println("Flags:")
	fmt.Println("  --data <path>       Training data (text file or .bin)")
	fmt.Println("  --dim <int>         Model dimension (default: 512)")
	fmt.Println("  --layers <int>      Number of layers (default: 6)")
	fmt.Println("  --heads <int>       Number of attention heads (default: 8)")
	fmt.Println("  --steps <int>       Training steps (default: 1000)")
	fmt.Println("  --lr <float>        Learning rate (default: 3e-4)")
	fmt.Println()
	fmt.Println("Engine: auto-detected (Metal MPSGraph / CUDA fused kernels / CPU fallback)")
	fmt.Println()
	fmt.Println("For fine-tuning with helix/needle, use: tesseract finetune --helix --needle")
	fmt.Fprintln(os.Stderr, "\nNot yet wired — GraphTrainEngine dispatch needs connection to CLI flags")
}

func cmdFinetune() {
	fmt.Println("tesseract finetune — fine-tune an existing model")
	fmt.Println()
	fmt.Println("Uses helix DNA optimizer for forward-only training. Optional needle INT8 kernels")
	fmt.Println("for training INT8 weights directly (no LoRA, no full-precision copies).")
	fmt.Println()
	fmt.Println("Usage: tesseract finetune <model> <data> [flags]")
	fmt.Println()
	fmt.Println("Flags:")
	fmt.Println("  --helix             Enable helix DNA optimizer (recommended)")
	fmt.Println("  --needle            Enable needle INT8 kernels (maximum performance)")
	fmt.Println("  --lr <float>        Learning rate (default: 1e-4)")
	fmt.Println("  --steps <int>       Fine-tuning steps (default: 500)")
	fmt.Println()
	fmt.Println("Default: helix forward-only + AdamW on auto-detected engine")
	fmt.Fprintln(os.Stderr, "\nNot yet wired — helix fine-tuning dispatch needs connection to CLI flags")
}

func cmdResume() {
	fmt.Println("tesseract resume — resume training from checkpoint")
	fmt.Println()
	fmt.Println("Usage: tesseract resume <checkpoint-dir> <data>")
	fmt.Fprintln(os.Stderr, "\nNot yet wired — checkpoint resume needs connection to CLI flags")
}
