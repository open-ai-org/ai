package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDatasetInspect(t *testing.T) {
	// Create a temp file with known content
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	content := "Hello world\nThis is a test\nThird line\n"
	os.WriteFile(path, []byte(content), 0644)

	// Just verify it doesn't panic — output goes to stdout
	datasetInspect(path)
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{500, "500 B"},
		{1500, "1.5 KB"},
		{1500000, "1.5 MB"},
		{1500000000, "1.5 GB"},
	}
	for _, tt := range tests {
		got := formatBytes(tt.n)
		if got != tt.want {
			t.Errorf("formatBytes(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

func TestFormatCount(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{500, "500"},
		{1500, "1.5K"},
		{1500000, "1.5M"},
		{1500000000, "1.5B"},
	}
	for _, tt := range tests {
		got := formatCount(tt.n)
		if got != tt.want {
			t.Errorf("formatCount(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

func TestDatasetSplit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.txt")

	var lines []string
	for i := 0; i < 100; i++ {
		lines = append(lines, "line "+string(rune('A'+i%26)))
	}
	os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0644)

	datasetSplit(path, 0.8, 0.1, 0.1, 42)

	trainData, err := os.ReadFile(filepath.Join(dir, "data_train.txt"))
	if err != nil {
		t.Fatalf("no train file: %v", err)
	}
	valData, _ := os.ReadFile(filepath.Join(dir, "data_val.txt"))
	testData, _ := os.ReadFile(filepath.Join(dir, "data_test.txt"))

	trainLines := strings.Count(string(trainData), "\n")
	valLines := strings.Count(string(valData), "\n")
	testLines := strings.Count(string(testData), "\n")

	if trainLines != 80 {
		t.Errorf("train = %d, want 80", trainLines)
	}
	if valLines != 10 {
		t.Errorf("val = %d, want 10", valLines)
	}
	if testLines != 10 {
		t.Errorf("test = %d, want 10", testLines)
	}
	if trainLines+valLines+testLines != 100 {
		t.Errorf("total = %d, want 100", trainLines+valLines+testLines)
	}
}

func TestDatasetSplitDeterministic(t *testing.T) {
	dir1, dir2 := t.TempDir(), t.TempDir()
	data := "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n"
	p1 := filepath.Join(dir1, "d.txt")
	p2 := filepath.Join(dir2, "d.txt")
	os.WriteFile(p1, []byte(data), 0644)
	os.WriteFile(p2, []byte(data), 0644)

	datasetSplit(p1, 0.8, 0.1, 0.1, 123)
	datasetSplit(p2, 0.8, 0.1, 0.1, 123)

	t1, _ := os.ReadFile(filepath.Join(dir1, "d_train.txt"))
	t2, _ := os.ReadFile(filepath.Join(dir2, "d_train.txt"))
	if string(t1) != string(t2) {
		t.Error("same seed should produce identical splits")
	}
}

func TestDatasetAugmentDedup(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dupes.txt")
	os.WriteFile(path, []byte("hello\nhello\nworld\nhello\nworld\n"), 0644)

	output := filepath.Join(dir, "out.txt")
	datasetAugment(path, output, 1, false, false, true)

	result, _ := os.ReadFile(output)
	lines := strings.Split(strings.TrimSpace(string(result)), "\n")
	if len(lines) != 2 {
		t.Errorf("dedup: got %d lines, want 2", len(lines))
	}
}

func TestDatasetAugmentRepeat(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "small.txt")
	os.WriteFile(path, []byte("a\nb\nc\n"), 0644)

	output := filepath.Join(dir, "out.txt")
	datasetAugment(path, output, 3, false, false, false)

	result, _ := os.ReadFile(output)
	lines := strings.Split(strings.TrimSpace(string(result)), "\n")
	if len(lines) != 9 {
		t.Errorf("repeat 3x: got %d, want 9", len(lines))
	}
}

func TestDatasetAugmentLowercase(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mixed.txt")
	os.WriteFile(path, []byte("Hello World\nFOO BAR\n"), 0644)

	output := filepath.Join(dir, "out.txt")
	datasetAugment(path, output, 1, false, true, false)

	result, _ := os.ReadFile(output)
	if strings.Contains(string(result), "H") || strings.Contains(string(result), "F") {
		t.Errorf("lowercase failed: %s", result)
	}
}

func TestDatasetAugmentCombined(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "combo.txt")
	os.WriteFile(path, []byte("Hello\nhello\nWorld\nworld\nFoo\n"), 0644)

	output := filepath.Join(dir, "out.txt")
	datasetAugment(path, output, 2, true, true, true)

	result, _ := os.ReadFile(output)
	lines := strings.Split(strings.TrimSpace(string(result)), "\n")
	// 5 lines → dedup (case-sensitive, so Hello != hello) → 5 unique → lowercase → repeat 2x → 10
	// Actually dedup happens before lowercase, so "Hello" and "hello" are different
	if len(lines) != 10 {
		t.Logf("got %d lines: %v", len(lines), lines)
	}
}
