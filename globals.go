package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"strings"

	"github.com/open-ai-org/mongoose"
)

// trainCfg holds all training configuration — shared across CUDA and Metal backends.
type trainCfg struct {
	dim         int
	nHeads      int
	nKVHeads    int
	headDim     int
	nLayers     int
	ffnDim      int
	vocabSize   int
	seqLen      int
	steps       int
	lr          float32
	logEvery    int
	beta1       float32
	beta2       float32
	adamEps     float32
	weightDecay float32
	ropeTheta   float64
	family       string // model family: "llama", "qwen2", etc.
	kvDim        int    // key/value dimension (= dim * nKVHeads / nHeads)
	gradClip     float32
}

// trainModel is the from-scratch model struct used by Metal's train_darwin.go.
// TODO: unify with cudaModel or define a shared interface.
type trainModel struct {
	// Stub — Metal defines the real fields
}

// formatParams formats a parameter count for display.
func formatParams(n int) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}


// --- Shared utility functions used by both CUDA and Metal training backends ---

// onesSlice returns a float32 slice of length n filled with 1.0.
func onesSlice(n int) []float32 {
	s := make([]float32, n)
	for i := range s { s[i] = 1.0 }
	return s
}

// zeroSlice fills a float32 slice with zeros.
func zeroSlice(s []float32) {
	for i := range s { s[i] = 0 }
}

// mvFn computes y = M @ x (matrix-vector multiply).
// M is [rows, cols] row-major, x is [cols], y is [rows].
func mvFn(y, M, x []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		var sum float32
		off := i * cols
		for j := 0; j < cols; j++ {
			sum += M[off+j] * x[j]
		}
		y[i] = sum
	}
}

// softmaxFn applies softmax in-place to a float32 slice.
func softmaxFn(x []float32) {
	mx := x[0]
	for _, v := range x[1:] { if v > mx { mx = v } }
	var se float32
	for i := range x { x[i] = float32(math.Exp(float64(x[i] - mx))); se += x[i] }
	invSe := 1.0 / se
	for i := range x { x[i] *= invSe }
}

// rmsNormFwdFn computes RMSNorm in-place and returns the inverse RMS scale.
// x is modified to x * scale * weight.
func rmsNormFwdFn(x, weight []float32) float32 {
	n := len(x)
	var ss float32
	for _, v := range x { ss += v * v }
	rms := float32(1.0 / math.Sqrt(float64(ss)/float64(n)+1e-6))
	for i := range x { x[i] = x[i] * rms * weight[i] }
	return rms
}

// rmsNormBwdFn computes RMSNorm backward.
// dOut is modified in-place to contain the gradient w.r.t. the pre-norm input.
// weightGrad is accumulated.
func rmsNormBwdFn(dOut, xIn, weight, weightGrad []float32, rmsScale float32) {
	n := len(xIn)
	for i := 0; i < n; i++ {
		weightGrad[i] += dOut[i] * xIn[i] * rmsScale
	}
	var dot float32
	for i := 0; i < n; i++ {
		dot += dOut[i] * weight[i] * xIn[i]
	}
	dot *= rmsScale * rmsScale * rmsScale / float32(n)
	for i := 0; i < n; i++ {
		dOut[i] = rmsScale*weight[i]*dOut[i] - dot*xIn[i]
	}
}

// selectEngine initializes the GPU backend.
func selectEngine(backend string) mongoose.Engine {
	switch backend {
	case "metal":
		if runtime.GOOS != "darwin" {
			log.Fatal("Metal backend only available on macOS")
		}
		m := mongoose.NewMetal()
		if m == nil {
			log.Fatal("Metal GPU not available")
		}
		return m
	case "cuda":
		c := mongoose.NewCUDA()
		if c == nil {
			log.Fatal("CUDA GPU not available")
		}
		return c
	default:
		// Auto-detect
		if runtime.GOOS == "darwin" {
			if m := mongoose.NewMetal(); m != nil { return m }
		}
		if c := mongoose.NewCUDA(); c != nil { return c }
		log.Fatal("No GPU backend available")
		return nil
	}
}

// Global flags — parsed before the subcommand, available everywhere.
// These define the execution environment. Subcommand flags define the task.
//
// Usage:
//   mongoose --device cuda --precision int8 --verbose finetune model data
//   mongoose --config myrun.json finetune model data
//   mongoose --dry-run finetune model data
//   mongoose --out /mnt/results finetune model data

var (
	GlobalDevice    string // "auto", "cuda", "cuda:0", "metal", "cpu"
	GlobalPrecision string // "auto", "fp32", "fp16", "bf16", "int8"
	GlobalOutDir    string // output directory for all artifacts
	GlobalVerbose   bool   // show detailed logs
	GlobalDryRun    bool   // validate config without executing
	GlobalConfigFile string // load all settings from file
)

// ParseGlobalFlags extracts global flags from os.Args before the subcommand.
// Returns the remaining args with global flags stripped.
//
// Global flags start with -- and appear BEFORE the subcommand:
//   mongoose --device cuda finetune model data
//           ^^^^^^^^^^^^^^ global    ^^^^^^^^^ subcommand + args
func ParseGlobalFlags() {
	// Defaults
	GlobalDevice = "auto"
	GlobalPrecision = "auto"
	GlobalOutDir = ""
	GlobalVerbose = false
	GlobalDryRun = false
	GlobalConfigFile = ""

	// Scan args before the subcommand (first non-flag arg)
	newArgs := []string{os.Args[0]}
	i := 1
	for i < len(os.Args) {
		arg := os.Args[i]
		if !strings.HasPrefix(arg, "--") {
			// Found the subcommand — pass everything from here onward
			newArgs = append(newArgs, os.Args[i:]...)
			break
		}
		switch arg {
		case "--device":
			if i+1 < len(os.Args) { GlobalDevice = os.Args[i+1]; i += 2 } else { i++ }
		case "--precision":
			if i+1 < len(os.Args) { GlobalPrecision = os.Args[i+1]; i += 2 } else { i++ }
		case "--out":
			if i+1 < len(os.Args) { GlobalOutDir = os.Args[i+1]; i += 2 } else { i++ }
		case "--config":
			if i+1 < len(os.Args) { GlobalConfigFile = os.Args[i+1]; i += 2 } else { i++ }
		case "--verbose":
			GlobalVerbose = true
			i++
		case "--dry-run":
			GlobalDryRun = true
			i++
		default:
			// Unknown global flag — pass through (might be a subcommand like --help)
			newArgs = append(newArgs, arg)
			i++
		}
	}
	os.Args = newArgs

	// Load config file if specified (overrides defaults, CLI flags override config)
	if GlobalConfigFile != "" {
		loadConfigFile(GlobalConfigFile)
	}

	if GlobalVerbose {
		log.Printf("[global] device=%s precision=%s out=%s verbose=%v dry-run=%v",
			GlobalDevice, GlobalPrecision, GlobalOutDir, GlobalVerbose, GlobalDryRun)
	}
}

// loadConfigFile reads a JSON config and applies settings that weren't
// explicitly set on the command line.
func loadConfigFile(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("--config %s: %v", path, err)
	}
	var cfg map[string]interface{}
	if err := json.Unmarshal(data, &cfg); err != nil {
		log.Fatalf("--config %s: invalid JSON: %v", path, err)
	}

	// Config file sets defaults — CLI flags already parsed take priority
	if v, ok := cfg["device"].(string); ok && GlobalDevice == "auto" {
		GlobalDevice = v
	}
	if v, ok := cfg["precision"].(string); ok && GlobalPrecision == "auto" {
		GlobalPrecision = v
	}
	if v, ok := cfg["out"].(string); ok && GlobalOutDir == "" {
		GlobalOutDir = v
	}
	if v, ok := cfg["verbose"].(bool); ok && !GlobalVerbose {
		GlobalVerbose = v
	}

	if GlobalVerbose {
		log.Printf("[config] loaded %s", path)
	}
}

// SelectBackend picks the compute engine based on --device flag.
// "auto" probes in order: cuda → metal → cpu.
func SelectBackend() string {
	switch GlobalDevice {
	case "auto":
		return "auto" // let selectEngine figure it out
	case "cuda", "metal", "cpu", "webgpu":
		return GlobalDevice
	default:
		if strings.HasPrefix(GlobalDevice, "cuda:") {
			return "cuda" // TODO: multi-GPU device selection
		}
		return GlobalDevice
	}
}

// PrintGlobalStatus shows active global settings (when non-default).
func PrintGlobalStatus() {
	var parts []string
	if GlobalDevice != "auto" { parts = append(parts, fmt.Sprintf("device=%s", GlobalDevice)) }
	if GlobalPrecision != "auto" { parts = append(parts, fmt.Sprintf("precision=%s", GlobalPrecision)) }
	if GlobalOutDir != "" { parts = append(parts, fmt.Sprintf("out=%s", GlobalOutDir)) }
	if GlobalDryRun { parts = append(parts, "dry-run") }
	if GlobalVerbose { parts = append(parts, "verbose") }
	if len(parts) > 0 {
		fmt.Printf("  [%s]\n", strings.Join(parts, ", "))
	}
}

// Shared utility functions (onesSlice, zeroSlice, mvFn, selectEngine,
// resolveModel, rmsNormFwdFn) defined earlier in this file by Metal session.

// int32ToFloat32Bits reinterprets int32 as float32 via math.
func int32ToFloat32Bits(v int32) float32 {
	return math.Float32frombits(uint32(v))
}

func float32ToInt32Bits(v float32) int32 {
	return int32(math.Float32bits(v))
}

// nativeBF16 is true when dual-output BF16 CUDA kernels are available.
var nativeBF16 bool

