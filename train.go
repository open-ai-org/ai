package main

import (
	"fmt"
	"os"
)

func cmdTrain() {
	fmt.Println("tesseract train — train a model from scratch")
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
	fmt.Println("  --helix             Use helix DNA optimizer")
	fmt.Println("  --needle            Use needle INT8 kernels")
	fmt.Println()
	fmt.Println("Default: AdamW optimizer on auto-detected engine (Metal/CUDA/CPU)")
	fmt.Fprintln(os.Stderr, "Training not yet wired — use GraphTrainEngine for fused dispatch")
}

func cmdFinetune() {
	fmt.Println("tesseract finetune — fine-tune an existing model")
	fmt.Println()
	fmt.Println("Usage: tesseract finetune <model> <data> [flags]")
	fmt.Fprintln(os.Stderr, "Fine-tuning not yet wired — coming soon")
}

func cmdResume() {
	fmt.Println("tesseract resume — resume training from checkpoint")
	fmt.Println()
	fmt.Println("Usage: tesseract resume <checkpoint-dir> <data>")
	fmt.Fprintln(os.Stderr, "Resume not yet wired — coming soon")
}
