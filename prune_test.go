package main

import (
	"math"
	"testing"
)

func TestPruneUnstructured(t *testing.T) {
	data := []float32{1.0, 0.1, -0.5, 0.01, 3.0, -0.02, 0.7, -0.001}

	pruneUnstructured(data, 0.5)

	nZeros := 0
	for _, v := range data {
		if v == 0 {
			nZeros++
		}
	}
	if nZeros != 4 {
		t.Errorf("50%% prune: got %d zeros, want 4", nZeros)
	}

	// Largest values should survive
	if data[0] != 1.0 {
		t.Errorf("1.0 should survive, got %f", data[0])
	}
	if data[4] != 3.0 {
		t.Errorf("3.0 should survive, got %f", data[4])
	}
}

func TestPruneUnstructuredHighSparsity(t *testing.T) {
	data := make([]float32, 100)
	for i := range data {
		data[i] = float32(i) * 0.01
	}

	pruneUnstructured(data, 0.9)

	nZeros := 0
	for _, v := range data {
		if v == 0 {
			nZeros++
		}
	}
	if nZeros != 90 {
		t.Errorf("90%% prune: got %d zeros, want 90", nZeros)
	}
}

func TestPruneStructured(t *testing.T) {
	rows, cols := 4, 3
	data := []float32{
		1, 2, 3,     // row 0: norm = sqrt(14) ≈ 3.74
		0.01, 0.01, 0.01, // row 1: norm ≈ 0.017 (smallest)
		5, 0, 0,     // row 2: norm = 5
		0.1, 0.1, 0.1,   // row 3: norm ≈ 0.17
	}

	pruneStructured(data, rows, cols, 0.5) // prune 2 of 4 rows

	// Row 1 and 3 should be zeroed (smallest norms)
	for c := 0; c < cols; c++ {
		if data[1*cols+c] != 0 {
			t.Errorf("row 1 should be zeroed, got %f at col %d", data[1*cols+c], c)
		}
		if data[3*cols+c] != 0 {
			t.Errorf("row 3 should be zeroed, got %f at col %d", data[3*cols+c], c)
		}
	}

	// Rows 0 and 2 should survive
	if data[0] != 1 {
		t.Errorf("row 0 should survive")
	}
	if data[2*cols] != 5 {
		t.Errorf("row 2 should survive")
	}
}

func TestPruneZeroSparsityNoOp(t *testing.T) {
	data := []float32{1, 2, 3}
	original := make([]float32, len(data))
	copy(original, data)

	pruneUnstructured(data, 0.0001) // essentially 0

	for i := range data {
		if data[i] != original[i] {
			t.Errorf("near-zero prune changed data[%d]: %f → %f", i, original[i], data[i])
		}
	}
}

func TestEmbeddingNorm(t *testing.T) {
	embed := []float32{3.0, 4.0, 0, 0, 0, 0} // dim=2, pos=0: norm should be 5
	norm := embeddingNorm(embed, 0, 2)
	if math.Abs(norm-5.0) > 1e-5 {
		t.Errorf("embeddingNorm = %f, want 5.0", norm)
	}
}

func TestCosineDist(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	d := cosineDist(a, b)
	if d > 1e-5 {
		t.Errorf("identical vectors: dist = %f, want 0", d)
	}

	c := []float32{0, 1, 0}
	d2 := cosineDist(a, c)
	if math.Abs(d2-1.0) > 1e-5 {
		t.Errorf("orthogonal vectors: dist = %f, want 1.0", d2)
	}
}
