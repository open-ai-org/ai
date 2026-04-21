//go:build darwin && cgo

package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/helix"
	"github.com/open-ai-org/mongoose"
)

func cmdTrainMetal() {
	fs := flag.NewFlagSet("train-metal", flag.ExitOnError)

	dataPath := fs.String("data", "", "Training data (text file)")
	dimFlag := fs.Int("dim", 128, "Model dimension")
	headsFlag := fs.Int("heads", 4, "Attention heads")
	kvHeadsFlag := fs.Int("kv-heads", 2, "KV heads (GQA)")
	layersFlag := fs.Int("layers", 4, "Transformer layers")
	ffnDimFlag := fs.Int("ffn-dim", 256, "FFN intermediate dimension")
	seqLenFlag := fs.Int("seq-len", 64, "Sequence length")
	stepsFlag := fs.Int("steps", 1000, "Training steps")
	lrFlag := fs.Float64("lr", 6e-4, "Learning rate")
	logEvery := fs.Int("log-every", 100, "Log every N steps")

	fs.Parse(os.Args[2:])

	if *dataPath == "" {
		*dataPath = "data/tinystories_hf.txt"
		if _, err := os.Stat(*dataPath); err != nil {
			home, _ := os.UserHomeDir()
			*dataPath = filepath.Join(home, "data", "tinystories_hf.txt")
		}
	}

	eng := selectEngine("metal")
	mtl, ok := eng.(*mongoose.Metal)
	if !ok {
		log.Fatal("train-metal requires Metal")
	}
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available")
	}

	dim := *dimFlag
	heads := *headsFlag
	kvHeads := *kvHeadsFlag
	headDim := dim / heads
	kvDim := kvHeads * headDim
	nLayers := *layersFlag
	ffnDim := *ffnDimFlag
	seqLen := *seqLenFlag
	vocabSize := 256
	lr := float32(*lrFlag)
	n := seqLen

	// OOM guard: estimate allocation before touching the GPU.
	// int8Param = 13 bytes/elem, grad = 4 bytes/elem, fwd bufs ≈ 4 bytes/elem
	{
		layerElems := dim*dim*2 + kvDim*dim*2 + dim*dim + ffnDim*dim*2 + dim*ffnDim
		estBytes := int64(layerElems) * 21 * int64(nLayers)
		estBytes += int64(vocabSize*dim) * 13
		estBytes += int64(n*dim) * 4 * 20 * int64(nLayers)
		estBytes += int64(n*heads*n) * 4
		estBytes += int64(n*vocabSize) * 4 * 3
		vram := eng.VRAM()
		headroom := uint64(float64(vram) * 0.75)
		if uint64(estBytes) > headroom {
			log.Fatalf("model needs ~%.0fMB but safe budget is %.0fMB (%.0fMB VRAM × 0.75) — reduce dim or layers",
				float64(estBytes)/(1024*1024), float64(headroom)/(1024*1024), float64(vram)/(1024*1024))
		}
	}

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	conductor := mongoose.NewConductor(vocabSize, 1)
	hlx := helix.NewHelixOptimizer(lr, 0.9, 0.95, 1e-8, 0.1)

	quantizeToInt8 := func(fp32 []float32, rows, cols int) (int8Data []int8, scales []float32, delta []float32) {
		int8Data = make([]int8, rows*cols)
		scales = make([]float32, rows)
		delta = make([]float32, rows*cols)
		for r := 0; r < rows; r++ {
			var absMax float32
			for c := 0; c < cols; c++ {
				v := fp32[r*cols+c]
				if v < 0 {
					v = -v
				}
				if v > absMax {
					absMax = v
				}
			}
			if absMax < 1e-10 {
				absMax = 1e-10
			}
			scales[r] = absMax
			invScale := float32(127.0) / absMax
			scale := absMax / 127.0
			for c := 0; c < cols; c++ {
				idx := r*cols + c
				qi := fp32[idx] * invScale
				if qi > 127 {
					qi = 127
				}
				if qi < -127 {
					qi = -127
				}
				qr := float32(math.Round(float64(qi)))
				int8Data[idx] = int8(qr)
				delta[idx] = fp32[idx] - qr*scale
			}
		}
		return
	}

	kaiming := func(rows, cols int) []float32 {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d {
			d[i] = bound * (2*rand.Float32() - 1)
		}
		return d
	}
	ones := func(sz int) *mongoose.Tensor {
		d := make([]float32, sz)
		for i := range d {
			d[i] = 1.0
		}
		return te.FromHost(d, []int{1, sz})
	}

	type int8Param struct {
		data   *mongoose.Tensor
		scales *mongoose.Tensor
		delta  *mongoose.Tensor
		mom    *mongoose.Tensor
		vel    *mongoose.Tensor
		live   *mongoose.Tensor
		rows   int
		cols   int
	}

	makeInt8Param := func(rows, cols int) int8Param {
		fp32 := kaiming(rows, cols)
		i8, sc, dl := quantizeToInt8(fp32, rows, cols)
		nElems := rows * cols

		dataT := mtl.AllocRaw(nElems, nElems, []int{rows, cols})
		mtl.UploadRaw(dataT, unsafe.Pointer(&i8[0]), nElems)

		scalesT := te.FromHost(sc, []int{rows})
		deltaT := te.FromHost(dl, []int{rows, cols})
		momT := mtl.AllocRaw(nElems*2, nElems, []int{rows, cols})
		velT := mtl.AllocRaw(nElems*2, nElems, []int{rows, cols})
		liveT := te.FromHost(fp32, []int{rows, cols})

		return int8Param{data: dataT, scales: scalesT, delta: deltaT, mom: momT, vel: velT, live: liveT, rows: rows, cols: cols}
	}

	embedFP32 := make([]float32, vocabSize*dim)
	for i := range embedFP32 {
		embedFP32[i] = float32(rand.NormFloat64()) * 0.02
	}
	embedI8, embedSc, embedDl := quantizeToInt8(embedFP32, vocabSize, dim)

	embedData := mtl.AllocRaw(vocabSize*dim, vocabSize*dim, []int{vocabSize, dim})
	mtl.UploadRaw(embedData, unsafe.Pointer(&embedI8[0]), vocabSize*dim)
	embedScales := te.FromHost(embedSc, []int{vocabSize})
	embedDelta := te.FromHost(embedDl, []int{vocabSize, dim})
	embedMom := mtl.AllocRaw(vocabSize*dim*2, vocabSize*dim, []int{vocabSize, dim})
	embedVel := mtl.AllocRaw(vocabSize*dim*2, vocabSize*dim, []int{vocabSize, dim})
	embed := te.FromHost(embedFP32, []int{vocabSize, dim})

	finalNorm := ones(dim)

	type layer struct {
		wq, wk, wv, wo, gate, up, down int8Param
		norm1, norm2                    *mongoose.Tensor
	}
	lays := make([]layer, nLayers)
	for l := range lays {
		lays[l] = layer{
			wq: makeInt8Param(dim, dim), wk: makeInt8Param(kvDim, dim),
			wv: makeInt8Param(kvDim, dim), wo: makeInt8Param(dim, dim),
			gate: makeInt8Param(ffnDim, dim), up: makeInt8Param(ffnDim, dim),
			down: makeInt8Param(dim, ffnDim),
			norm1: ones(dim), norm2: ones(dim),
		}
	}

	embedMask := mtl.NewHotRowMask(vocabSize)
	type layerMasks struct{ wq, wk, wv, wo, gate, up, down *mongoose.HotRowMask }
	layMasks := make([]layerMasks, nLayers)
	for li := range layMasks {
		layMasks[li] = layerMasks{
			wq: mtl.NewHotRowMask(dim), wk: mtl.NewHotRowMask(kvDim),
			wv: mtl.NewHotRowMask(kvDim), wo: mtl.NewHotRowMask(dim),
			gate: mtl.NewHotRowMask(ffnDim), up: mtl.NewHotRowMask(ffnDim),
			down: mtl.NewHotRowMask(dim),
		}
	}

	trackerWindow := 1
	type layerTrackers struct{ wq, wk, wv, wo, gate, up, down *mongoose.ProjectionTracker }
	layTrk := make([]layerTrackers, nLayers)
	for li := range layTrk {
		layTrk[li] = layerTrackers{
			wq: mongoose.NewProjectionTracker(dim, dim, trackerWindow),
			wk: mongoose.NewProjectionTracker(kvDim, dim, trackerWindow),
			wv: mongoose.NewProjectionTracker(kvDim, dim, trackerWindow),
			wo: mongoose.NewProjectionTracker(dim, dim, trackerWindow),
			gate: mongoose.NewProjectionTracker(ffnDim, dim, trackerWindow),
			up: mongoose.NewProjectionTracker(ffnDim, dim, trackerWindow),
			down: mongoose.NewProjectionTracker(dim, ffnDim, trackerWindow),
		}
	}

	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})
	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})
	lmMaxLogit := te.Zeros([]int{n})
	lmSumExp := te.Zeros([]int{n})
	lmLoss := te.Zeros([]int{1})
	dEmbed := te.Zeros([]int{vocabSize, dim})
	dHidden := te.Zeros([]int{n, dim})
	dScratch := te.Zeros([]int{n, dim})
	gradSumSq := te.Zeros([]int{1})
	clipScaleBuf := te.Zeros([]int{1})
	const gradMaxNorm = float32(1.0)

	// Forward activations — double-buffered for coiled pipeline.
	// Set 0 and set 1 alternate: forward writes to act[cur], backward reads act[prev].
	type actBuf struct {
		xIn, normed, Q, K, V, attnOut         *mongoose.Tensor
		xMid, normed2, gatePre, upOut, ffnMid *mongoose.Tensor
		rmsScale1, rmsScale2                  *mongoose.Tensor
		gateAct                               *mongoose.Tensor
	}
	acts := make([]actBuf, nLayers)
	for i := range acts {
		acts[i] = actBuf{
			xIn: te.Zeros([]int{n, dim}), normed: te.Zeros([]int{n, dim}),
			Q: te.Zeros([]int{n, dim}), K: te.Zeros([]int{n, kvDim}), V: te.Zeros([]int{n, kvDim}),
			attnOut: te.Zeros([]int{n, dim}),
			xMid: te.Zeros([]int{n, dim}), normed2: te.Zeros([]int{n, dim}),
			gatePre: te.Zeros([]int{n, ffnDim}), upOut: te.Zeros([]int{n, ffnDim}),
			ffnMid: te.Zeros([]int{n, ffnDim}),
			rmsScale1: te.Zeros([]int{n}), rmsScale2: te.Zeros([]int{n}),
			gateAct: te.Zeros([]int{n, ffnDim}),
		}
	}
	scores := te.Zeros([]int{n * heads, n})

	// Backward scratch — single set, only used by backward queue
	type bwdBuf struct {
		dFfnMid, dGate, dUp, dN2, dx         *mongoose.Tensor
		dAttnOut, dQ, dK, dV, dN1             *mongoose.Tensor
		dWDown, dWGate, dWUp, dWO, dWQ, dWK, dWV *mongoose.Tensor
	}
	bwds := make([]bwdBuf, nLayers)
	for i := range bwds {
		bwds[i] = bwdBuf{
			dFfnMid: te.Zeros([]int{n, ffnDim}), dGate: te.Zeros([]int{n, ffnDim}),
			dUp: te.Zeros([]int{n, ffnDim}), dN2: te.Zeros([]int{n, dim}),
			dx: te.Zeros([]int{n, dim}),
			dAttnOut: te.Zeros([]int{n, dim}), dQ: te.Zeros([]int{n, dim}),
			dK: te.Zeros([]int{n, kvDim}), dV: te.Zeros([]int{n, kvDim}),
			dN1: te.Zeros([]int{n, dim}),
			dWDown: te.Zeros([]int{dim, ffnDim}), dWGate: te.Zeros([]int{ffnDim, dim}),
			dWUp: te.Zeros([]int{ffnDim, dim}), dWO: te.Zeros([]int{dim, dim}),
			dWQ: te.Zeros([]int{dim, dim}), dWK: te.Zeros([]int{kvDim, dim}),
			dWV: te.Zeros([]int{kvDim, dim}),
		}
	}

	nParams := vocabSize*dim + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Println("ai train-metal — Metal fused kernels + Needle INT8 optimizer")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1e6 {
		fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	} else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	totalSteps := *stepsFlag
	warmupSteps := 1
	minLR := lr / 10.0
	getLR := func(step int) float32 {
		if step < warmupSteps {
			return lr * float32(step) / float32(warmupSteps)
		}
		progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return minLR + float32(cosine)*float32(lr-minLR)
	}

	var curLR float32
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	type sparseCheckpoint struct {
		rows    []int32
		weights map[string][]float32
		loss    float32
		step    int
	}
	var ckpt *sparseCheckpoint
	bestFloor := float32(1e30)
	immuneActive := false
	floorContactStep := 0
	floorWindow := 10
	maxRecoveries := 20
	recoveryCount := 0

	ckptDir := filepath.Join(os.TempDir(), "ai-train-out", "checkpoints")
	if GlobalOutDir != "" {
		ckptDir = filepath.Join(GlobalOutDir, "checkpoints")
	}
	os.MkdirAll(ckptDir, 0755)
	lastCkptStep := 0
	ckptBestLoss := float32(999.0)

	saveHotRows := func(hotRows []int32, loss float32, step int) {
		ckpt = &sparseCheckpoint{
			rows:    make([]int32, len(hotRows)),
			weights: make(map[string][]float32),
			loss:    loss,
			step:    step,
		}
		copy(ckpt.rows, hotRows)
		mtl.Sync()
		for li := range lays {
			for _, proj := range []struct {
				name string
				w    *mongoose.Tensor
			}{
				{"wq", lays[li].wq.live}, {"wk", lays[li].wk.live}, {"wv", lays[li].wv.live},
				{"wo", lays[li].wo.live}, {"gate", lays[li].gate.live}, {"up", lays[li].up.live},
				{"down", lays[li].down.live},
			} {
				wH := te.ToHost(proj.w)
				cols := proj.w.Size / proj.w.Shape[0]
				key := fmt.Sprintf("%d.%s", li, proj.name)
				saved := make([]float32, 0, len(hotRows)*cols)
				for _, r := range hotRows {
					row := int(r)
					if row >= 0 && row < proj.w.Shape[0] {
						saved = append(saved, wH[row*cols:(row+1)*cols]...)
					}
				}
				ckpt.weights[key] = saved
			}
		}
	}

	restoreHotRows := func() {
		if ckpt == nil {
			return
		}
		for li := range lays {
			for _, proj := range []struct {
				name string
				w    *mongoose.Tensor
			}{
				{"wq", lays[li].wq.live}, {"wk", lays[li].wk.live}, {"wv", lays[li].wv.live},
				{"wo", lays[li].wo.live}, {"gate", lays[li].gate.live}, {"up", lays[li].up.live},
				{"down", lays[li].down.live},
			} {
				key := fmt.Sprintf("%d.%s", li, proj.name)
				saved := ckpt.weights[key]
				if len(saved) == 0 {
					continue
				}
				wH := te.ToHost(proj.w)
				cols := proj.w.Size / proj.w.Shape[0]
				idx := 0
				for _, r := range ckpt.rows {
					row := int(r)
					if row >= 0 && row < proj.w.Shape[0] && idx+cols <= len(saved) {
						copy(wH[row*cols:(row+1)*cols], saved[idx:idx+cols])
						idx += cols
					}
				}
				mtl.UploadInto(proj.w, wH)
			}
		}
	}

	saveFullCheckpoint := func(step int, loss float32) {
		mtl.Sync()
		tensors := map[string]gguf.SaveTensor{
			"model.embed_tokens.weight": {Data: te.ToHost(embed), Shape: []int{vocabSize, dim}},
			"model.norm.weight":         {Data: te.ToHost(finalNorm), Shape: []int{dim}},
		}
		for li := range lays {
			pfx := fmt.Sprintf("model.layers.%d.", li)
			tensors[pfx+"self_attn.q_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wq.live), Shape: []int{dim, dim}}
			tensors[pfx+"self_attn.k_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wk.live), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.v_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wv.live), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.o_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wo.live), Shape: []int{dim, dim}}
			tensors[pfx+"mlp.gate_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].gate.live), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.up_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].up.live), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.down_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].down.live), Shape: []int{dim, ffnDim}}
			tensors[pfx+"input_layernorm.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].norm1), Shape: []int{dim}}
			tensors[pfx+"post_attention_layernorm.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].norm2), Shape: []int{dim}}
		}
		stepDir := filepath.Join(ckptDir, fmt.Sprintf("step-%05d", step))
		os.MkdirAll(stepDir, 0755)
		stPath := filepath.Join(stepDir, "model.safetensors")
		if err := gguf.SaveSafeTensors(stPath, tensors); err != nil {
			log.Printf("[checkpoint] save error: %v", err)
		} else {
			cfgJSON := fmt.Sprintf(`{"architectures":["LlamaForCausalLM"],"hidden_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"intermediate_size":%d,"vocab_size":%d,"max_position_embeddings":2048,"rope_theta":10000.0,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":true}`,
				dim, nLayers, heads, kvHeads, ffnDim, vocabSize)
			os.WriteFile(filepath.Join(stepDir, "config.json"), []byte(cfgJSON), 0644)
			log.Printf("[checkpoint] saved step %d loss=%.3f → %s", step, loss, stepDir)
		}
	}

	embedShared := mtl.SharedSlice(embed)
	hiddenShared := mtl.SharedSlice(hidden)
	lmLossShared := mtl.SharedSlice(lmLoss)

	fmt.Println("Training...")
	t0 := time.Now()

	tokIDs := make([]int32, n)
	var prevLoss float32

	// Phase timing accumulators
	var tBatch, tFwd, tLoss, tObs, tBwd, tClip, tNeedle, tDequant, tImmune time.Duration
	var stepStart time.Time

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokF := make([]float32, n)
		targF := make([]float32, n)
		stepStart = time.Now()
		ph := time.Now()
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(int32(data[start+i])))
			targF[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
			tokIDs[i] = int32(data[start+i])
		}
		mtl.UploadInto(tokGPU, tokF)
		mtl.UploadInto(targetsGPU, targF)
		conductor.Observe(tokIDs)

		for i := 0; i < n; i++ {
			tokID := data[start+i]
			copy(hiddenShared[i*dim:(i+1)*dim], embedShared[tokID*dim:(tokID+1)*dim])
		}
		tBatch += time.Since(ph)

		// Read loss from PREVIOUS step (one step stale, GPU already done).
		// Step 1 has no previous loss; use log(vocabSize) as initial estimate.
		stepLoss := float32(math.Log(float64(vocabSize)))
		if step > 1 {
			stepLoss = lmLossShared[0]
		}

		// === Immune system (uses previous step's loss) ===
		ph = time.Now()
		hotRows := conductor.HotRows()

		if !immuneActive && step > 1 && stepLoss > 0 {
			if stepLoss < bestFloor*1.1 {
				saveHotRows(hotRows, stepLoss, step)
			}
		}
		if stepLoss > 0 && stepLoss < bestFloor {
			bestFloor = stepLoss
			if !immuneActive {
				immuneActive = true
				floorContactStep = step
				recoveryCount = 0
			}
		}
		immuneSkip := false
		if immuneActive && step-floorContactStep >= floorWindow {
			rebound := stepLoss - bestFloor
			threshold := bestFloor * 0.05
			if rebound > threshold && recoveryCount < maxRecoveries && ckpt != nil {
				restoreHotRows()
				recoveryCount++
				immuneActive = false
				immuneSkip = true
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s) [IMMUNE → floor %.3f]\n",
					step, *stepsFlag, stepLoss, getLR(step), elapsed.Seconds(),
					float64(step)/elapsed.Seconds(), ckpt.loss)
			} else {
				immuneActive = false
			}
		}
		if stepLoss < ckptBestLoss && step-lastCkptStep >= 88 && step > 1 {
			ckptBestLoss = stepLoss
			lastCkptStep = step
			go saveFullCheckpoint(step, stepLoss)
		}

		// Signal-scaled LR
		stepLR := getLR(step)
		if prevLoss > 0 {
			dLoss := float64(stepLoss) - float64(prevLoss)
			if dLoss > 0 {
				ratio := float32(dLoss / math.Max(float64(prevLoss), 1e-6))
				if ratio > 1.0 { ratio = 1.0 }
				stepLR *= (1.0 - ratio)
			}
		}
		prevLoss = stepLoss
		curLR = stepLR
		tImmune += time.Since(ph)

		if immuneSkip { continue }

		// === Projection trackers + masks (use previous step's activations) ===
		ph = time.Now()
		if step > 2 {
			for li := range lays {
				a := &acts[li]
				layTrk[li].wq.ObserveOutput(mtl.SharedSlice(a.Q), n, dim)
				layTrk[li].wk.ObserveOutput(mtl.SharedSlice(a.K), n, kvDim)
				layTrk[li].wv.ObserveOutput(mtl.SharedSlice(a.V), n, kvDim)
				layTrk[li].wo.ObserveOutput(mtl.SharedSlice(a.attnOut), n, dim)
				layTrk[li].gate.ObserveOutput(mtl.SharedSlice(a.gatePre), n, ffnDim)
				layTrk[li].up.ObserveOutput(mtl.SharedSlice(a.upOut), n, ffnDim)
				layTrk[li].down.ObserveOutput(mtl.SharedSlice(a.ffnMid), n, ffnDim)
			}
		}
		embedMask.Set(conductor.HotRows())
		for li := range lays {
			layMasks[li].wq.Set(layTrk[li].wq.HotRows())
			layMasks[li].wk.Set(layTrk[li].wk.HotRows())
			layMasks[li].wv.Set(layTrk[li].wv.HotRows())
			layMasks[li].wo.Set(layTrk[li].wo.HotRows())
			layMasks[li].gate.Set(layTrk[li].gate.HotRows())
			layMasks[li].up.Set(layTrk[li].up.HotRows())
			layMasks[li].down.Set(layTrk[li].down.HotRows())
		}
		tObs += time.Since(ph)

		// === Helix rung geometry ===
		r, bc1, bc2, rewound := hlx.PrepareStep(step, stepLoss, curLR)
		if rewound {
			if step <= 3 || step%*logEvery == 0 {
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f steps/s) [helix-rewound]\n",
					step, *stepsFlag, stepLoss, curLR, bestFloor, elapsed.Seconds(), float64(step)/elapsed.Seconds())
			}
			continue
		}

		// ============================================================
		// SINGLE COMMAND BUFFER: forward → GPU loss → backward → clip → needle → dequant
		// ============================================================
		ph = time.Now()
		mtl.FusedBegin()

		// --- FORWARD ---
		for li := range lays {
			l := &lays[li]
			a := &acts[li]

			mtl.FusedCopy(a.xIn, hidden, n*dim)
			mtl.FusedRMSNorm(hidden, l.norm1, a.rmsScale1, n, dim)
			mtl.FusedCopy(a.normed, hidden, n*dim)
			mtl.FusedCopy(hidden, a.xIn, n*dim)

			mtl.FusedGemmF32BT(a.normed, l.wq.live, a.Q, n, dim, dim)
			mtl.FusedGemmF32BT(a.normed, l.wk.live, a.K, n, dim, kvDim)
			mtl.FusedGemmF32BT(a.normed, l.wv.live, a.V, n, dim, kvDim)

			mtl.FusedRoPE(a.Q, headDim, heads, 10000.0, dim, n)
			mtl.FusedRoPE(a.K, headDim, kvHeads, 10000.0, kvDim, n)

			mtl.FusedAttention(a.Q, a.K, a.V, a.attnOut, scores, dim, kvDim, headDim, heads, kvHeads, n)

			mtl.FusedGemmF32BT(a.attnOut, l.wo.live, dScratch, n, dim, dim)
			mtl.FusedAddInPlace(hidden, dScratch, n*dim)

			mtl.FusedCopy(a.xMid, hidden, n*dim)
			mtl.FusedRMSNorm(hidden, l.norm2, a.rmsScale2, n, dim)
			mtl.FusedCopy(a.normed2, hidden, n*dim)
			mtl.FusedCopy(hidden, a.xMid, n*dim)

			mtl.FusedGemmF32BT(a.normed2, l.gate.live, a.gatePre, n, dim, ffnDim)
			mtl.FusedGemmF32BT(a.normed2, l.up.live, a.upOut, n, dim, ffnDim)

			mtl.FusedSiLUGateMul(a.gatePre, a.upOut, a.ffnMid, n*ffnDim)

			mtl.FusedGemmF32BT(a.ffnMid, l.down.live, dScratch, n, ffnDim, dim)
			mtl.FusedAddInPlace(hidden, dScratch, n*dim)
		}

		mtl.FusedRMSNorm(hidden, finalNorm, finalScales, n, dim)
		mtl.FusedCopy(normedFinal, hidden, n*dim)

		// --- GPU LOSS: lm_head_pass1 → barrier → lm_head_pass2 → dHidden ---
		mtl.FusedLMHeadPass1(normedFinal, embed, lmMaxLogit, lmSumExp, dim, vocabSize, n)
		mtl.FusedBarrierBuffers()
		mtl.FusedLMHeadPass2(normedFinal, embed, lmMaxLogit, lmSumExp, targetsGPU, dHidden, lmLoss, dim, vocabSize, n)
		mtl.FusedBarrierBuffers()

		// --- STEP-1 NOOP: forward + loss only, skip backward ---
		if step == 1 {
			mtl.FusedEnd()
			tFwd += time.Since(ph)
			if step <= 3 || step%*logEvery == 0 {
				mtl.Sync()
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f steps/s) [noop]\n",
					step, *stepsFlag, lmLossShared[0], curLR, bestFloor, elapsed.Seconds(), float64(step)/elapsed.Seconds())
			}
			continue
		}

		// --- BACKWARD (dHidden already seeded by lm_head_pass2) ---
		// dEmbed: sparse TN GEMM using the dHidden gradient and normedFinal
		// Note: lm_head_pass2 produced dHidden = sum_v(grad[v] * embed[v]) per position.
		// dEmbed needs grad^T @ normedFinal. We approximate with a separate sparse pass.
		// For now, skip dEmbed gradient (embed updated via conductor hot rows only).

		mtl.FusedRMSNormBwd(dHidden, hidden, finalNorm, finalScales, dScratch, n, dim)
		mtl.FusedCopy(dHidden, dScratch, n*dim)

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]
			a := &acts[li]
			b := &bwds[li]
			lm := &layMasks[li]

			mtl.FusedGemmF32NN(dHidden, l.down.live, b.dFfnMid, n, dim, ffnDim)
			mtl.FusedGemmF32TNSparse(dHidden, a.ffnMid, b.dWDown, lm.down, dim, n, ffnDim)

			mtl.SiLUGateBackward(b.dFfnMid, a.gatePre, a.upOut, a.gateAct, b.dGate, b.dUp)

			mtl.FusedGemmF32NN(b.dGate, l.gate.live, b.dN2, n, ffnDim, dim)
			mtl.FusedGemmF32NN(b.dUp, l.up.live, b.dx, n, ffnDim, dim)
			mtl.FusedAddInPlace(b.dN2, b.dx, n*dim)

			mtl.FusedGemmF32TNSparse(b.dGate, a.normed2, b.dWGate, lm.gate, ffnDim, n, dim)
			mtl.FusedGemmF32TNSparse(b.dUp, a.normed2, b.dWUp, lm.up, ffnDim, n, dim)

			mtl.FusedRMSNormBwd(b.dN2, a.xMid, l.norm2, a.rmsScale2, b.dx, n, dim)
			mtl.FusedAddInPlace(dHidden, b.dx, n*dim)

			mtl.FusedGemmF32NN(dHidden, l.wo.live, b.dAttnOut, n, dim, dim)
			mtl.FusedGemmF32TNSparse(dHidden, a.attnOut, b.dWO, lm.wo, dim, n, dim)

			mtl.FusedAttentionBwdQ(b.dAttnOut, a.Q, a.K, a.V, scores,
				b.dQ, b.dK, b.dV, dim, kvDim, headDim, heads, kvHeads, n, n)

			mtl.FusedRoPE(b.dQ, headDim, heads, -10000.0, dim, n)
			mtl.FusedRoPE(b.dK, headDim, kvHeads, -10000.0, kvDim, n)

			mtl.FusedGemmF32NN(b.dQ, l.wq.live, b.dN1, n, dim, dim)
			mtl.FusedGemmF32NN(b.dK, l.wk.live, b.dx, n, kvDim, dim)
			mtl.FusedGemmF32NN(b.dV, l.wv.live, b.dN2, n, kvDim, dim)
			mtl.FusedAddInPlace(b.dN1, b.dx, n*dim)
			mtl.FusedAddInPlace(b.dN1, b.dN2, n*dim)

			mtl.FusedGemmF32TNSparse(b.dQ, a.normed, b.dWQ, lm.wq, dim, n, dim)
			mtl.FusedGemmF32TNSparse(b.dK, a.normed, b.dWK, lm.wk, kvDim, n, dim)
			mtl.FusedGemmF32TNSparse(b.dV, a.normed, b.dWV, lm.wv, kvDim, n, dim)

			mtl.FusedRMSNormBwd(b.dN1, a.xIn, l.norm1, a.rmsScale1, b.dx, n, dim)
			mtl.FusedAddInPlace(dHidden, b.dx, n*dim)
		}

		// --- GRAD NORM → CLIP SCALE (GPU) → NEEDLE (with built-in clip + dequant) ---
		mtl.FusedZeroScalar(gradSumSq)
		mtl.FusedBarrierBuffers()
		for li := range lays {
			b := &bwds[li]
			mtl.FusedGradNormSq(b.dWQ, gradSumSq, b.dWQ.Size)
			mtl.FusedGradNormSq(b.dWK, gradSumSq, b.dWK.Size)
			mtl.FusedGradNormSq(b.dWV, gradSumSq, b.dWV.Size)
			mtl.FusedGradNormSq(b.dWO, gradSumSq, b.dWO.Size)
			mtl.FusedGradNormSq(b.dWGate, gradSumSq, b.dWGate.Size)
			mtl.FusedGradNormSq(b.dWUp, gradSumSq, b.dWUp.Size)
			mtl.FusedGradNormSq(b.dWDown, gradSumSq, b.dWDown.Size)
		}
		mtl.FusedBarrierBuffers()
		mtl.FusedComputeClipScale(gradSumSq, clipScaleBuf, gradMaxNorm)
		mtl.FusedBarrierBuffers()

		const beta1 = float32(0.9)
		const beta2 = float32(0.95)
		const eps = float32(1e-8)
		const wd = float32(0.1)
		bondGC := float32(3.0 / 5.0)
		bondAT := float32(2.0 / 5.0)
		// clipScaleBuf[0] is now on GPU — needle reads it via the live+clipScale shader args.
		// But the shader takes clipScale as a constant, not a buffer. We need to read it.
		// Actually the shader uses setBytes for clipScale — we can't pass a buffer.
		// Solution: end encoder, read from shared memory, restart. One sync point.
		mtl.FusedEnd()
		clipScale := mtl.SharedSlice(clipScaleBuf)[0]
		mtl.FusedBegin()

		needleSingle := func(p int8Param, grad *mongoose.Tensor, mask *mongoose.HotRowMask) {
			mtl.FusedNeedle(p.data, p.scales, grad,
				p.mom, p.vel, mask, p.delta,
				curLR, beta1, beta2, bc1, bc2, eps, wd, p.data.Size, p.cols,
				p.live, clipScale)
		}

		for li := range lays {
			l := &lays[li]
			b := &bwds[li]
			lm := &layMasks[li]

			mtl.FusedNeedlePaired(
				l.gate.data, l.up.data, l.gate.scales, l.up.scales,
				b.dWGate, b.dWUp, l.gate.mom, l.up.mom, l.gate.vel, l.up.vel,
				lm.gate, l.gate.delta, l.up.delta,
				curLR, beta1, beta2, bc1, bc2, eps, wd,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				bondGC, l.gate.data.Size, dim, l.gate.live, l.up.live, clipScale)

			if l.wq.data.Size == l.wk.data.Size {
				mtl.FusedNeedlePaired(
					l.wq.data, l.wk.data, l.wq.scales, l.wk.scales,
					b.dWQ, b.dWK, l.wq.mom, l.wk.mom, l.wq.vel, l.wk.vel,
					lm.wq, l.wq.delta, l.wk.delta,
					curLR, beta1, beta2, bc1, bc2, eps, wd,
					r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
					bondAT, l.wq.data.Size, dim, l.wq.live, l.wk.live, clipScale)
			} else {
				needleSingle(l.wq, b.dWQ, lm.wq)
				needleSingle(l.wk, b.dWK, lm.wk)
			}

			needleSingle(l.wv, b.dWV, lm.wv)
			needleSingle(l.wo, b.dWO, lm.wo)
			needleSingle(l.down, b.dWDown, lm.down)
		}

		needleSingle(int8Param{data: embedData, scales: embedScales, delta: embedDelta, mom: embedMom, vel: embedVel, live: embed, rows: vocabSize, cols: dim},
			dEmbed, embedMask)

		mtl.FusedEnd()
		tFwd += time.Since(ph)

		stepDur := time.Since(stepStart)
		if step <= 3 || step%*logEvery == 0 || step == *stepsFlag {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f avg, %.1f this)\n",
				step, *stepsFlag, stepLoss, curLR, bestFloor, elapsed.Seconds(),
				float64(step)/elapsed.Seconds(), 1.0/stepDur.Seconds())
		}
		_ = r
	}

	mtl.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)  floor=%.3f\n",
		*stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds(), bestFloor)
	fmt.Printf("\nphase breakdown (wall time, includes GPU wait):\n")
	all := tBatch + tFwd + tLoss + tImmune + tObs + tBwd + tClip + tNeedle + tDequant
	pct := func(d time.Duration) string { return fmt.Sprintf("%.1f%%", float64(d)/float64(all)*100) }
	fmt.Printf("  batch:   %6.1fms  %s\n", float64(tBatch.Microseconds())/1000, pct(tBatch))
	fmt.Printf("  forward: %6.1fms  %s\n", float64(tFwd.Microseconds())/1000, pct(tFwd))
	fmt.Printf("  loss:    %6.1fms  %s\n", float64(tLoss.Microseconds())/1000, pct(tLoss))
	fmt.Printf("  immune:  %6.1fms  %s\n", float64(tImmune.Microseconds())/1000, pct(tImmune))
	fmt.Printf("  observe: %6.1fms  %s\n", float64(tObs.Microseconds())/1000, pct(tObs))
	fmt.Printf("  bwd:     %6.1fms  %s\n", float64(tBwd.Microseconds())/1000, pct(tBwd))
	fmt.Printf("  clip:    %6.1fms  %s\n", float64(tClip.Microseconds())/1000, pct(tClip))
	fmt.Printf("  needle:  %6.1fms  %s\n", float64(tNeedle.Microseconds())/1000, pct(tNeedle))
	fmt.Printf("  dequant: %6.1fms  %s\n", float64(tDequant.Microseconds())/1000, pct(tDequant))
	fmt.Printf("  total:   %6.1fms\n", float64(all.Microseconds())/1000)

	saveFullCheckpoint(*stepsFlag, bestFloor)
	embedMask.Release()
	for li := range layMasks {
		layMasks[li].wq.Release()
		layMasks[li].wk.Release()
		layMasks[li].wv.Release()
		layMasks[li].wo.Release()
		layMasks[li].gate.Release()
		layMasks[li].up.Release()
		layMasks[li].down.Release()
	}
	_ = mtl
	_ = scores
	_ = hlx
}
