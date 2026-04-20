package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResumeFindsLatestCheckpoint(t *testing.T) {
	dir := t.TempDir()

	// Create checkpoint dirs with meta.json
	for _, name := range []string{"step-00100", "step-00200", "step-00300"} {
		ckptDir := filepath.Join(dir, name)
		os.MkdirAll(ckptDir, 0755)
		writeCkptMeta(ckptDir, &ckptMeta{
			Step: 100,
			Loss: 3.5,
			LR:   6e-4,
		})
	}

	latest := findLatestCheckpoint(dir)
	if filepath.Base(latest) != "step-00300" {
		t.Errorf("latest = %s, want step-00300", filepath.Base(latest))
	}
}

func TestResumeReadsCheckpointMeta(t *testing.T) {
	dir := t.TempDir()
	ckptDir := filepath.Join(dir, "step-00500")
	os.MkdirAll(ckptDir, 0755)

	writeCkptMeta(ckptDir, &ckptMeta{
		Step: 500,
		Loss: 2.1,
		LR:   3e-4,
	})

	meta, err := readCkptMeta(ckptDir)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Step != 500 {
		t.Errorf("step = %d, want 500", meta.Step)
	}
	if meta.Loss != 2.1 {
		t.Errorf("loss = %f, want 2.1", meta.Loss)
	}
	if meta.LR != 3e-4 {
		t.Errorf("lr = %f, want 3e-4", meta.LR)
	}
}

func TestResumeCheckpointHasConfig(t *testing.T) {
	dir := t.TempDir()
	ckptDir := filepath.Join(dir, "step-01000")
	os.MkdirAll(ckptDir, 0755)

	cfg := map[string]interface{}{
		"hidden_size":         float64(128),
		"num_hidden_layers":   float64(4),
		"num_attention_heads": float64(4),
		"intermediate_size":   float64(256),
		"vocab_size":          float64(256),
	}
	data, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(filepath.Join(ckptDir, "config.json"), data, 0644)

	// Verify config can be read back
	readBack, err := os.ReadFile(filepath.Join(ckptDir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var parsed map[string]interface{}
	json.Unmarshal(readBack, &parsed)

	if parsed["hidden_size"].(float64) != 128 {
		t.Errorf("hidden_size = %v, want 128", parsed["hidden_size"])
	}
	if parsed["num_hidden_layers"].(float64) != 4 {
		t.Errorf("layers = %v, want 4", parsed["num_hidden_layers"])
	}
}

func TestWriteCkptMetaCreatesFile(t *testing.T) {
	dir := t.TempDir()
	err := writeCkptMeta(dir, &ckptMeta{Step: 42, Loss: 1.5, LR: 1e-4})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(filepath.Join(dir, "meta.json")); err != nil {
		t.Error("meta.json not created")
	}

	meta, err := readCkptMeta(dir)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Step != 42 {
		t.Errorf("step = %d, want 42", meta.Step)
	}
}

func TestFindLatestCheckpointEmpty(t *testing.T) {
	dir := t.TempDir()
	result := findLatestCheckpoint(dir)
	if result != "" {
		t.Errorf("expected empty, got %s", result)
	}
}

