package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/open-ai-org/gguf"
)

func TestOpenModelSafeTensors(t *testing.T) {
	dir := t.TempDir()

	// Create minimal safetensors model
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: make([]float32, 64*16), Shape: []int{64, 16}},
	}
	gguf.SaveSafeTensors(filepath.Join(dir, "model.safetensors"), tensors)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"hidden_size":16,"num_hidden_layers":1}`), 0644)

	m, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Format() != "safetensors" {
		t.Errorf("format = %s, want safetensors", m.Format())
	}
	if m.ConfigInt("hidden_size", 0) != 16 {
		t.Errorf("hidden_size = %d, want 16", m.ConfigInt("hidden_size", 0))
	}
	if !m.HasTensor("model.embed_tokens.weight") {
		t.Error("missing embed_tokens tensor")
	}

	data, err := m.ReadTensorFloat32("model.embed_tokens.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 64*16 {
		t.Errorf("tensor size = %d, want %d", len(data), 64*16)
	}
}

func TestOpenModelGGUF(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")

	// Create minimal GGUF file
	w := gguf.NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddUint32("llama.block_count", 1)
	w.AddUint32("llama.embedding_length", 16)
	w.AddTensorF32("token_embd.weight", make([]float32, 64*16), 64, 16)
	if err := w.Write(path); err != nil {
		t.Fatal(err)
	}

	m, err := OpenModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Format() != "gguf" {
		t.Errorf("format = %s, want gguf", m.Format())
	}
}

func TestOpenModelGGUFInDir(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")

	w := gguf.NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddTensorF32("token_embd.weight", make([]float32, 16), 1, 16)
	w.Write(path)

	m, err := OpenModel(dir) // pass directory, not file
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Format() != "gguf" {
		t.Errorf("format = %s, want gguf", m.Format())
	}
}

func TestOpenModelNotFound(t *testing.T) {
	_, err := OpenModel("/tmp/nonexistent_model_path_xyz")
	if err == nil {
		t.Error("expected error for nonexistent path")
	}
}

func TestOpenModelEmptyDir(t *testing.T) {
	dir := t.TempDir()
	_, err := OpenModel(dir)
	if err == nil {
		t.Error("expected error for empty directory")
	}
}

func TestHasSafeTensors(t *testing.T) {
	dir := t.TempDir()
	if hasSafeTensors(dir) {
		t.Error("empty dir should not have safetensors")
	}

	os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte{}, 0644)
	if !hasSafeTensors(dir) {
		t.Error("should detect safetensors file")
	}
}

func TestConfigHelpers(t *testing.T) {
	cfg := map[string]interface{}{
		"hidden_size": float64(256),
		"rope_theta":  float64(10000.0),
		"hidden_act":  "silu",
	}

	if configInt(cfg, "hidden_size", 0) != 256 {
		t.Error("configInt failed")
	}
	if configInt(cfg, "missing", 42) != 42 {
		t.Error("configInt default failed")
	}
	if configFloat(cfg, "rope_theta", 0) != 10000.0 {
		t.Error("configFloat failed")
	}
	if configInt(nil, "anything", 99) != 99 {
		t.Error("nil config should return default")
	}
}

func TestGGUFNameMapRoundTrip(t *testing.T) {
	nameMap := buildGGUFNameMap(2)

	// Check key mappings
	if nameMap["model.embed_tokens.weight"] != "token_embd.weight" {
		t.Errorf("embed mapping wrong: %s", nameMap["model.embed_tokens.weight"])
	}
	if nameMap["model.layers.0.self_attn.q_proj.weight"] != "blk.0.attn_q.weight" {
		t.Errorf("q_proj mapping wrong: %s", nameMap["model.layers.0.self_attn.q_proj.weight"])
	}
	if nameMap["model.layers.1.mlp.gate_proj.weight"] != "blk.1.ffn_gate.weight" {
		t.Errorf("gate mapping wrong: %s", nameMap["model.layers.1.mlp.gate_proj.weight"])
	}
}
