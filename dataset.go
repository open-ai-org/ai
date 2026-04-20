package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// cmdDataset handles dataset subcommands.
//
//	ai dataset inspect <file>
func cmdDataset(args map[string]string) {
	sub := args["_0"]
	file := args["_1"]
	if file == "" {
		file = args["file"]
	}

	switch sub {
	case "inspect":
		if file == "" {
			fmt.Fprintln(os.Stderr, "Usage: ai dataset inspect <file>")
			os.Exit(1)
		}
		datasetInspect(file)
	case "split":
		if file == "" {
			fmt.Fprintln(os.Stderr, "Usage: ai dataset split <file> [--train 0.8] [--val 0.1] [--test 0.1] [--seed 42]")
			os.Exit(1)
		}
		trainRatio := 0.8
		valRatio := 0.1
		seed := int64(42)
		if v, ok := args["train"]; ok { fmt.Sscanf(v, "%f", &trainRatio) }
		if v, ok := args["val"]; ok { fmt.Sscanf(v, "%f", &valRatio) }
		if v, ok := args["seed"]; ok { fmt.Sscanf(v, "%d", &seed) }
		testRatio := 1.0 - trainRatio - valRatio
		datasetSplit(file, trainRatio, valRatio, testRatio, seed)
	case "augment":
		if file == "" {
			fmt.Fprintln(os.Stderr, "Usage: ai dataset augment <file> [--output <file>] [--repeat N] [--shuffle] [--lowercase] [--dedup]")
			os.Exit(1)
		}
		output := args["output"]
		if output == "" { output = args["_2"] }
		repeat := 1
		if v, ok := args["repeat"]; ok { fmt.Sscanf(v, "%d", &repeat) }
		_, doShuffle := args["shuffle"]
		_, doLower := args["lowercase"]
		_, doDedup := args["dedup"]
		datasetAugment(file, output, repeat, doShuffle, doLower, doDedup)
	default:
		fmt.Fprintln(os.Stderr, "Usage:")
		fmt.Fprintln(os.Stderr, "  ai dataset inspect <file>              Preview dataset statistics")
		fmt.Fprintln(os.Stderr, "  ai dataset split <file> [--train 0.8]  Split into train/val/test")
		fmt.Fprintln(os.Stderr, "  ai dataset augment <file> [flags]      Apply transforms")
		os.Exit(1)
	}
}

// datasetInspect analyzes a text dataset and reports statistics.
func datasetInspect(path string) {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read: %v", err)
	}

	totalBytes := len(raw)
	totalLines := 0
	totalWords := 0
	lineLengths := make([]int, 0)
	byteFreq := make(map[byte]int)
	inWord := false
	lineLen := 0

	for _, b := range raw {
		byteFreq[b]++
		if b == '\n' {
			totalLines++
			lineLengths = append(lineLengths, lineLen)
			lineLen = 0
			inWord = false
		} else {
			lineLen++
			if b == ' ' || b == '\t' {
				inWord = false
			} else if !inWord {
				inWord = true
				totalWords++
			}
		}
	}
	if lineLen > 0 {
		totalLines++
		lineLengths = append(lineLengths, lineLen)
	}

	sort.Ints(lineLengths)
	var avgLine float64
	if len(lineLengths) > 0 {
		total := 0
		for _, l := range lineLengths {
			total += l
		}
		avgLine = float64(total) / float64(len(lineLengths))
	}

	uniqueBytes := len(byteFreq)

	// Estimate token count (rough: ~4 chars per token for English)
	estTokens := totalBytes / 4

	fmt.Printf("ai dataset inspect — %s\n\n", path)
	fmt.Printf("  size:        %s\n", formatBytes(totalBytes))
	fmt.Printf("  lines:       %d\n", totalLines)
	fmt.Printf("  words:       %d\n", totalWords)
	fmt.Printf("  unique bytes: %d / 256\n", uniqueBytes)
	fmt.Printf("  est. tokens: ~%s (byte-level: %s)\n", formatCount(estTokens), formatCount(totalBytes))
	fmt.Println()

	if len(lineLengths) > 0 {
		p50 := lineLengths[len(lineLengths)/2]
		p95 := lineLengths[int(float64(len(lineLengths))*0.95)]
		p99 := lineLengths[int(float64(len(lineLengths))*0.99)]
		fmt.Printf("  line length:\n")
		fmt.Printf("    min:  %d\n", lineLengths[0])
		fmt.Printf("    avg:  %.0f\n", avgLine)
		fmt.Printf("    p50:  %d\n", p50)
		fmt.Printf("    p95:  %d\n", p95)
		fmt.Printf("    p99:  %d\n", p99)
		fmt.Printf("    max:  %d\n", lineLengths[len(lineLengths)-1])
	}
	fmt.Println()

	// Recommended training config
	fmt.Printf("  Recommended:\n")
	if totalBytes < 1_000_000 {
		fmt.Printf("    seq_len:  64\n")
		fmt.Printf("    steps:    %d\n", totalBytes/64*10)
	} else if totalBytes < 100_000_000 {
		fmt.Printf("    seq_len:  128\n")
		fmt.Printf("    steps:    %d\n", min(totalBytes/128*3, 50000))
	} else {
		fmt.Printf("    seq_len:  256\n")
		fmt.Printf("    steps:    50000\n")
	}
}

func formatBytes(b int) string {
	switch {
	case b >= 1_000_000_000:
		return fmt.Sprintf("%.1f GB", float64(b)/1e9)
	case b >= 1_000_000:
		return fmt.Sprintf("%.1f MB", float64(b)/1e6)
	case b >= 1_000:
		return fmt.Sprintf("%.1f KB", float64(b)/1e3)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func formatCount(n int) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// datasetSplit partitions a text file into train/val/test splits.
// Splits by lines with optional shuffling. Writes to <base>_train.txt, etc.
func datasetSplit(path string, trainR, valR, testR float64, seed int64) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) != "" {
			lines = append(lines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("scan: %v", err)
	}

	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(lines), func(i, j int) { lines[i], lines[j] = lines[j], lines[i] })

	n := len(lines)
	nTrain := int(float64(n) * trainR)
	nVal := int(float64(n) * valR)
	nTest := n - nTrain - nVal

	trainLines := lines[:nTrain]
	valLines := lines[nTrain : nTrain+nVal]
	testLines := lines[nTrain+nVal:]

	ext := filepath.Ext(path)
	base := strings.TrimSuffix(path, ext)

	writeLines := func(name string, data []string) {
		outPath := base + "_" + name + ext
		out, err := os.Create(outPath)
		if err != nil {
			log.Fatalf("create %s: %v", outPath, err)
		}
		w := bufio.NewWriter(out)
		for _, line := range data {
			w.WriteString(line)
			w.WriteByte('\n')
		}
		w.Flush()
		out.Close()
		fmt.Printf("  %s: %d lines → %s\n", name, len(data), outPath)
	}

	fmt.Printf("ai dataset split — %s\n", path)
	fmt.Printf("  total lines: %d (seed=%d)\n", n, seed)
	fmt.Printf("  ratios: train=%.0f%% val=%.0f%% test=%.0f%%\n", trainR*100, valR*100, testR*100)
	fmt.Println()

	writeLines("train", trainLines)
	writeLines("val", valLines)
	if nTest > 0 {
		writeLines("test", testLines)
	}
	fmt.Println("\ndone.")
}

// datasetAugment applies text transformations to a dataset.
// Transforms: shuffle lines, lowercase, deduplicate, repeat N times.
func datasetAugment(path, output string, repeat int, shuffle, lowercase, dedup bool) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("scan: %v", err)
	}

	origCount := len(lines)
	fmt.Printf("ai dataset augment — %s\n", path)
	fmt.Printf("  input: %d lines\n", origCount)

	// Dedup
	if dedup {
		seen := make(map[string]bool)
		var unique []string
		for _, line := range lines {
			key := strings.TrimSpace(line)
			if key == "" {
				continue
			}
			if !seen[key] {
				seen[key] = true
				unique = append(unique, line)
			}
		}
		fmt.Printf("  dedup: %d → %d lines (-%d)\n", len(lines), len(unique), len(lines)-len(unique))
		lines = unique
	}

	// Lowercase
	if lowercase {
		for i := range lines {
			lines[i] = strings.ToLower(lines[i])
		}
		fmt.Println("  lowercase: applied")
	}

	// Repeat
	if repeat > 1 {
		var expanded []string
		for r := 0; r < repeat; r++ {
			expanded = append(expanded, lines...)
		}
		fmt.Printf("  repeat: %dx (%d → %d lines)\n", repeat, len(lines), len(expanded))
		lines = expanded
	}

	// Shuffle
	if shuffle {
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		rng.Shuffle(len(lines), func(i, j int) { lines[i], lines[j] = lines[j], lines[i] })
		fmt.Println("  shuffle: applied")
	}

	// Output
	if output == "" {
		ext := filepath.Ext(path)
		base := strings.TrimSuffix(path, ext)
		output = base + "_augmented" + ext
	}

	out, err := os.Create(output)
	if err != nil {
		log.Fatalf("create: %v", err)
	}
	w := bufio.NewWriter(out)
	for _, line := range lines {
		w.WriteString(line)
		w.WriteByte('\n')
	}
	w.Flush()
	out.Close()

	fmt.Printf("\n  output: %d lines → %s\n", len(lines), output)
	fmt.Println("done.")
}
