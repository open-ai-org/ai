package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// TODO: rewrite checkpoints for needle model (no LoRA adapters).
// Needle checkpoints save: modified INT8 weights + sparse momentum/velocity + mask state.

// findLatestCheckpoint finds the most recent checkpoint directory.
func findLatestCheckpoint(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			dirs = append(dirs, e.Name())
		}
	}
	if len(dirs) == 0 {
		return ""
	}
	sort.Strings(dirs)
	return filepath.Join(dir, dirs[len(dirs)-1])
}

// listCheckpoints returns sorted checkpoint directories.
func listCheckpoints(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			dirs = append(dirs, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(dirs)
	return dirs
}
