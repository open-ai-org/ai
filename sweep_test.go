package main

import (
	"testing"
)

func TestParseFloatList(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"1e-4,3e-4,6e-4", 3},
		{"0.001", 1},
		{"", 0}, // returns default (nil)
	}
	for _, tt := range tests {
		def := []float64(nil)
		result := parseFloatList(tt.input, def)
		if tt.input == "" {
			if result != nil {
				t.Errorf("empty input should return nil default")
			}
		} else if len(result) != tt.want {
			t.Errorf("parseFloatList(%q) = %d values, want %d", tt.input, len(result), tt.want)
		}
	}
}

func TestParseIntList(t *testing.T) {
	result := parseIntList("64,128,256", nil)
	if len(result) != 3 {
		t.Errorf("got %d, want 3", len(result))
	}
	if result[0] != 64 || result[1] != 128 || result[2] != 256 {
		t.Errorf("got %v, want [64 128 256]", result)
	}

	result2 := parseIntList("", []int{42})
	if len(result2) != 1 || result2[0] != 42 {
		t.Errorf("empty should return default, got %v", result2)
	}
}

func TestSweepTrialCPU(t *testing.T) {
	// Create minimal data
	data := make([]int, 1000)
	for i := range data {
		data[i] = i % 256
	}

	eng := selectEngine("cpu")
	loss, elapsed := runSweepTrialCPU(eng, data, 1e-3, 32, 2, 1, 2, 64, 256, 32, 50)

	if loss <= 0 || loss > 100 {
		t.Errorf("unexpected loss: %f", loss)
	}
	if elapsed <= 0 {
		t.Error("elapsed should be > 0")
	}
	t.Logf("CPU sweep trial: loss=%.4f elapsed=%v", loss, elapsed)
}

func TestRmsNormCPU(t *testing.T) {
	data := []float32{1, 2, 3, 4}
	weight := []float32{1, 1, 1, 1}
	rmsNormCPU(data, weight)

	// After RMSNorm, the vector should have approximately unit RMS
	var ss float32
	for _, v := range data {
		ss += v * v
	}
	rms := ss / float32(len(data))
	// Should be close to 1.0
	if rms < 0.5 || rms > 2.0 {
		t.Errorf("RMS after norm = %f, expected near 1.0", rms)
	}
}

func TestMatvec(t *testing.T) {
	// 2x3 matrix @ 3-vector = 2-vector
	mat := []float32{1, 2, 3, 4, 5, 6}
	vec := []float32{1, 1, 1}
	out := make([]float32, 2)
	matvec(out, mat, vec, 2, 3)

	if out[0] != 6 { // 1+2+3
		t.Errorf("out[0] = %f, want 6", out[0])
	}
	if out[1] != 15 { // 4+5+6
		t.Errorf("out[1] = %f, want 15", out[1])
	}
}
