package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/open-ai-org/gguf"
)

// ModelSource abstracts over SafeTensors directories and GGUF files.
// All tensor reads return float32 regardless of source format.
type ModelSource struct {
	format   string // "safetensors" or "gguf"
	st       *gguf.SafeTensors
	gr       *gguf.GGUFReader
	nameMap  map[string]string // HF name → GGUF name (only for GGUF)
	config   map[string]interface{}
	dir      string // directory containing the model (for tokenizer, config)
	nLayers  int
}

// OpenModel auto-detects and opens a model from a path.
// Supports: SafeTensors directory, single .gguf file, directory containing .gguf.
func OpenModel(path string) (*ModelSource, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("not found: %s", path)
	}

	// Single GGUF file
	if !info.IsDir() && strings.HasSuffix(path, ".gguf") {
		return openGGUFModel(path)
	}

	// Directory — check for safetensors first, then GGUF
	if info.IsDir() {
		// SafeTensors?
		if hasSafeTensors(path) {
			return openSafeTensorsModel(path)
		}
		// GGUF file in directory?
		if ggufPath := findGGUFInDir(path); ggufPath != "" {
			return openGGUFModel(ggufPath)
		}
		return nil, fmt.Errorf("no model files found in %s (need .safetensors or .gguf)", path)
	}

	return nil, fmt.Errorf("unsupported model path: %s", path)
}

func hasSafeTensors(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") || e.Name() == "model.safetensors.index.json" {
			return true
		}
	}
	return false
}

func findGGUFInDir(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".gguf") {
			return filepath.Join(dir, e.Name())
		}
	}
	return ""
}

func openSafeTensorsModel(dir string) (*ModelSource, error) {
	st, err := gguf.OpenSafeTensors(dir)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}

	cfg := loadConfig(dir)
	nLayers := configInt(cfg, "num_hidden_layers", 0)

	return &ModelSource{
		format:  "safetensors",
		st:      st,
		config:  cfg,
		dir:     dir,
		nLayers: nLayers,
	}, nil
}

func openGGUFModel(path string) (*ModelSource, error) {
	gr, err := gguf.OpenGGUF(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf: %w", err)
	}

	// Try to read config from GGUF metadata
	cfg := make(map[string]interface{})

	// Standard GGUF metadata keys
	if v := gr.MetadataUint32("llama.embedding_length"); v > 0 {
		cfg["hidden_size"] = float64(v)
	}
	if v := gr.MetadataUint32("llama.block_count"); v > 0 {
		cfg["num_hidden_layers"] = float64(v)
	}
	if v := gr.MetadataUint32("llama.attention.head_count"); v > 0 {
		cfg["num_attention_heads"] = float64(v)
	}
	if v := gr.MetadataUint32("llama.attention.head_count_kv"); v > 0 {
		cfg["num_key_value_heads"] = float64(v)
	}
	if v := gr.MetadataUint32("llama.feed_forward_length"); v > 0 {
		cfg["intermediate_size"] = float64(v)
	}
	if v := gr.MetadataUint32("llama.vocab_size"); v > 0 {
		cfg["vocab_size"] = float64(v)
	}
	if v := gr.MetadataUint32("llama.context_length"); v > 0 {
		cfg["max_position_embeddings"] = float64(v)
	}
	if v := gr.MetadataFloat32("llama.rope.freq_base"); v > 0 {
		cfg["rope_theta"] = float64(v)
	}
	if v := gr.MetadataFloat32("llama.attention.layer_norm_rms_epsilon"); v > 0 {
		cfg["rms_norm_eps"] = float64(v)
	}

	// Also check for config.json alongside the GGUF file
	dir := filepath.Dir(path)
	if fileCfg := loadConfig(dir); fileCfg != nil {
		for k, v := range fileCfg {
			if _, exists := cfg[k]; !exists {
				cfg[k] = v
			}
		}
	}

	nLayers := configInt(cfg, "num_hidden_layers", 0)

	// Build name map: HF → GGUF
	nameMap := buildGGUFNameMap(nLayers)

	return &ModelSource{
		format:  "gguf",
		gr:      gr,
		nameMap: nameMap,
		config:  cfg,
		dir:     dir,
		nLayers: nLayers,
	}, nil
}

// ReadTensorFloat32 reads a tensor by HuggingFace name, auto-translating for GGUF.
// Always returns float32 regardless of storage format (dequantizes Q8/Q4/F16).
func (m *ModelSource) ReadTensorFloat32(name string) ([]float32, error) {
	switch m.format {
	case "safetensors":
		data, _, err := m.st.ReadTensorFloat32(name)
		return data, err
	case "gguf":
		ggufName := name
		if mapped, ok := m.nameMap[name]; ok {
			ggufName = mapped
		}
		data, _, err := m.gr.ReadTensorFloat32(ggufName)
		if err != nil {
			// Try original name as fallback
			data, _, err = m.gr.ReadTensorFloat32(name)
		}
		return data, err
	default:
		return nil, fmt.Errorf("unsupported format: %s", m.format)
	}
}

// HasTensor checks if a tensor exists by HF name.
func (m *ModelSource) HasTensor(name string) bool {
	switch m.format {
	case "safetensors":
		return m.st.HasTensor(name)
	case "gguf":
		ggufName := name
		if mapped, ok := m.nameMap[name]; ok {
			ggufName = mapped
		}
		return m.gr.HasTensor(ggufName) || m.gr.HasTensor(name)
	default:
		return false
	}
}

// Config returns the model configuration.
func (m *ModelSource) Config() map[string]interface{} { return m.config }

// Format returns "safetensors" or "gguf".
func (m *ModelSource) Format() string { return m.format }

// Dir returns the directory containing the model (for tokenizer files).
func (m *ModelSource) Dir() string { return m.dir }

// Close releases resources.
func (m *ModelSource) Close() {
	if m.gr != nil {
		m.gr.Close()
	}
}

// ConfigInt reads an int from the config with a default.
func (m *ModelSource) ConfigInt(key string, def int) int {
	return configInt(m.config, key, def)
}

// ConfigFloat reads a float from the config with a default.
func (m *ModelSource) ConfigFloat(key string, def float64) float64 {
	return configFloat(m.config, key, def)
}

// ConfigString reads a string from the config with a default.
func (m *ModelSource) ConfigString(key string, def string) string {
	if v, ok := m.config[key].(string); ok {
		return v
	}
	return def
}

// helpers

func loadConfig(dir string) map[string]interface{} {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil
	}
	var cfg map[string]interface{}
	json.Unmarshal(data, &cfg)
	return cfg
}

func configInt(cfg map[string]interface{}, key string, def int) int {
	if cfg == nil {
		return def
	}
	if v, ok := cfg[key].(float64); ok {
		return int(v)
	}
	return def
}

func configFloat(cfg map[string]interface{}, key string, def float64) float64 {
	if cfg == nil {
		return def
	}
	if v, ok := cfg[key].(float64); ok {
		return v
	}
	return def
}

// ResolveAndOpen resolves a model name and opens it.
// Handles: direct paths, model names, Ollama-style names.
func ResolveAndOpen(name string) *ModelSource {
	// Direct GGUF file path
	if strings.HasSuffix(name, ".gguf") {
		if _, err := os.Stat(name); err == nil {
			m, err := OpenModel(name)
			if err != nil {
				log.Fatalf("open model: %v", err)
			}
			return m
		}
	}

	// Try resolveModel (searches ~/.ai/models, etc.)
	path := resolveModel(name)
	m, err := OpenModel(path)
	if err != nil {
		log.Fatalf("open model %s: %v", path, err)
	}
	return m
}
