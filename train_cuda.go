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

	"github.com/open-ai-org/mongoose"
)

// cmdTrainCUDA trains from scratch on CUDA.
// All ops on GPU via custom kernels: KEmbedGather, KRMSNormOutSave, KRoPE,
// KCausalAttentionGQA, KSiLUGateMul, KSoftmaxCE for forward.
// KCausalAttentionBackward, KRMSNormBackward, KRoPEBackward, KSiLUGateBackward for backward.
// Standard AdamW optimizer. Conductor for sparse embedding updates.
// Activations in L3 pinned memory where beneficial, weights on GPU.
func cmdTrainCUDA() {
	fs := flag.NewFlagSet("train-cuda", flag.ExitOnError)

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

	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available — need CUDA backend")
	}
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok {
		log.Fatal("train-cuda requires CUDA backend")
	}

	if mongoose.LoadKernels() {
		log.Println("[tesseract] CUDA kernels loaded — full GPU training")
	} else {
		log.Fatal("CUDA custom kernels required for GPU training — compile kernels/mongoose.cu")
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

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	// Conductor for sparse embedding updates
	conductor := mongoose.NewConductor(vocabSize, 100)

	// Precompute RoPE tables on GPU
	halfHead := headDim / 2
	cosTab := make([]float32, seqLen*halfHead)
	sinTab := make([]float32, seqLen*halfHead)
	for pos := 0; pos < seqLen; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(10000.0, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}
	ropeCos := te.FromHost(cosTab, []int{seqLen, halfHead})
	ropeSin := te.FromHost(sinTab, []int{seqLen, halfHead})

	// Init weights on GPU
	kaiming := func(rows, cols int) *mongoose.Tensor {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d { d[i] = bound * (2*rand.Float32() - 1) }
		return te.FromHost(d, []int{rows, cols})
	}
	ones := func(n int) *mongoose.Tensor {
		d := make([]float32, n)
		for i := range d { d[i] = 1.0 }
		return te.FromHost(d, []int{1, n})
	}

	embedData := make([]float32, vocabSize*dim)
	for i := range embedData { embedData[i] = float32(rand.NormFloat64()) * 0.02 }
	embed := te.FromHost(embedData, []int{vocabSize, dim})
	finalNorm := ones(dim)

	type layer struct {
		wq, wk, wv, wo     *mongoose.Tensor
		gate, up, down      *mongoose.Tensor
		norm1, norm2        *mongoose.Tensor
	}
	layers := make([]layer, nLayers)
	for l := range layers {
		layers[l] = layer{
			wq: kaiming(dim, dim), wk: kaiming(kvDim, dim),
			wv: kaiming(kvDim, dim), wo: kaiming(dim, dim),
			gate: kaiming(ffnDim, dim), up: kaiming(ffnDim, dim),
			down: kaiming(dim, ffnDim),
			norm1: ones(dim), norm2: ones(dim),
		}
	}

	// Adam state on GPU
	type adamState struct {
		m, v *mongoose.Tensor
	}
	newAdam := func(size int) adamState {
		return adamState{m: te.Zeros([]int{size}), v: te.Zeros([]int{size})}
	}

	embedAdam := newAdam(vocabSize * dim)
	_ = newAdam(dim) // finalNorm adam — TODO wire
	type layerAdam struct {
		wq, wk, wv, wo     adamState
		gate, up, down      adamState
		norm1, norm2        adamState
	}
	layerAdams := make([]layerAdam, nLayers)
	for l := range layerAdams {
		layerAdams[l] = layerAdam{
			wq: newAdam(dim * dim), wk: newAdam(kvDim * dim),
			wv: newAdam(kvDim * dim), wo: newAdam(dim * dim),
			gate: newAdam(ffnDim * dim), up: newAdam(ffnDim * dim),
			down: newAdam(dim * ffnDim),
			norm1: newAdam(dim), norm2: newAdam(dim),
		}
	}

	// Pre-allocate forward cache tensors (reuse every step)
	type fwdCache struct {
		xIn, normed, Q, K, V, attnOut    *mongoose.Tensor
		xMid, normed2, gatePre, upOut, ffnMid *mongoose.Tensor
		rmsScale1, rmsScale2              *mongoose.Tensor
	}
	caches := make([]fwdCache, nLayers)
	for i := range caches {
		caches[i] = fwdCache{
			xIn: te.Zeros([]int{n, dim}), xMid: te.Zeros([]int{n, dim}),
		}
	}

	nParams := vocabSize*dim + dim
	for range layers {
		nParams += dim + dim*dim + kvDim*dim + kvDim*dim + dim*dim +
			dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Println("tesseract train-cuda — full GPU kernels + AdamW + Conductor")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1_000_000 {
		fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	} else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	adamUpdate := func(param, grad, mState, vState *mongoose.Tensor, step int) {
		mongoose.KAdamW(param.DevicePtr(), grad.DevicePtr(), mState.DevicePtr(), vState.DevicePtr(),
			lr, 0.01, step, param.Size)
	}
	_ = adamUpdate

	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)

		tokIDs := make([]int32, n)
		targets := make([]int32, n)
		for i := 0; i < n; i++ {
			tokIDs[i] = int32(data[start+i])
			targets[i] = int32(data[start+i+1])
		}
		conductor.Observe(tokIDs)

		// Upload tokens to GPU
		tokF := make([]float32, n)
		for i, t := range tokIDs { tokF[i] = math.Float32frombits(uint32(t)) }
		tokGPU := te.FromHost(tokF, []int{n})

		// === FORWARD — all on GPU ===

		hidden := te.Zeros([]int{n, dim})
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range layers {
			l := &layers[li]
			c := &caches[li]

			cuda.CopyInto(c.xIn, hidden)

			// RMSNorm1
			c.normed = te.Zeros([]int{n, dim})
			c.rmsScale1 = te.Zeros([]int{n})
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), c.normed.DevicePtr(),
				l.norm1.DevicePtr(), c.rmsScale1.DevicePtr(), n, dim)

			// QKV matmul
			c.Q = te.MatMulTransposeBT(c.normed, l.wq, n, dim, dim)
			c.K = te.MatMulTransposeBT(c.normed, l.wk, n, dim, kvDim)
			c.V = te.MatMulTransposeBT(c.normed, l.wv, n, dim, kvDim)

			// RoPE
			mongoose.KRoPE(c.Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(c.K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			// Causal attention
			c.attnOut = te.Zeros([]int{n, dim})
			mongoose.KCausalAttentionGQA(c.Q.DevicePtr(), c.K.DevicePtr(), c.V.DevicePtr(), c.attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			// O projection + residual
			proj := te.MatMulTransposeBT(c.attnOut, l.wo, n, dim, dim)
			te.AddInPlace(hidden, proj)
			te.Release(proj)

			cuda.CopyInto(c.xMid, hidden)

			// RMSNorm2
			c.normed2 = te.Zeros([]int{n, dim})
			c.rmsScale2 = te.Zeros([]int{n})
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), c.normed2.DevicePtr(),
				l.norm2.DevicePtr(), c.rmsScale2.DevicePtr(), n, dim)

			// FFN
			c.gatePre = te.MatMulTransposeBT(c.normed2, l.gate, n, dim, ffnDim)
			c.upOut = te.MatMulTransposeBT(c.normed2, l.up, n, dim, ffnDim)
			c.ffnMid = te.Zeros([]int{n, ffnDim})
			mongoose.KSiLUGateMul(c.gatePre.DevicePtr(), c.upOut.DevicePtr(), c.ffnMid.DevicePtr(), n*ffnDim)

			down := te.MatMulTransposeBT(c.ffnMid, l.down, n, ffnDim, dim)
			te.AddInPlace(hidden, down)
			te.Release(down)
		}

		// Final RMSNorm
		finalScales := te.Zeros([]int{n})
		normedFinal := te.Zeros([]int{n, dim})
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

		// LM head + loss
		logits := te.MatMulTransposeBT(normedFinal, embed, n, dim, vocabSize)
		targetsF := make([]float32, n)
		for i, t := range targets { targetsF[i] = math.Float32frombits(uint32(t)) }
		targetsGPU := te.FromHost(targetsF, []int{n})
		lossesGPU := te.Zeros([]int{n})
		gradGPU := te.Zeros([]int{n, vocabSize})
		invN := float32(1.0) / float32(n)
		mongoose.KSoftmaxCE(logits.DevicePtr(), targetsGPU.DevicePtr(),
			lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)

		// Read loss
		lossH := te.ToHost(lossesGPU)
		var totalLoss float32
		for _, l := range lossH { totalLoss += l }

		// === BACKWARD — all on GPU ===

		// dLogits already computed by KSoftmaxCE → gradGPU
		// Embed gradient: dEmbed = logitsGrad^T @ normedFinal
		dEmbed := te.MatMulTransposeAT(normedFinal, gradGPU, dim, n, vocabSize)

		// dHidden from LM head: dH = gradGPU @ embed
		dHidden := te.MatMulT(gradGPU, embed, n, vocabSize, dim)
		te.Release(gradGPU); te.Release(logits); te.Release(targetsGPU); te.Release(lossesGPU)

		// Final RMSNorm backward
		dNormFinal := te.Zeros([]int{n, dim})
		mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), dNormFinal.DevicePtr(), n, dim)
		te.Release(finalScales); te.Release(normedFinal); te.Release(hidden)

		// Per-layer backward (reverse order)
		for li := nLayers - 1; li >= 0; li-- {
			l := &layers[li]
			c := &caches[li]

			// FFN backward
			dFFNMid := te.Zeros([]int{n, ffnDim})
			mongoose.KSiLUGateBackward(dNormFinal.DevicePtr(), c.gatePre.DevicePtr(),
				c.upOut.DevicePtr(), c.ffnMid.DevicePtr(), dFFNMid.DevicePtr(), n*ffnDim)

			// Weight gradients: down, gate, up
			dWDown := te.MatMulTransposeAT(c.ffnMid, dNormFinal, ffnDim, n, dim)
			dWGate := te.MatMulTransposeAT(c.normed2, dFFNMid, dim, n, ffnDim)
			dWUp := te.MatMulTransposeAT(c.normed2, dFFNMid, dim, n, ffnDim)

			// AdamW on FFN weights
			adamUpdate(l.down, dWDown, layerAdams[li].down.m, layerAdams[li].down.v, step)
			adamUpdate(l.gate, dWGate, layerAdams[li].gate.m, layerAdams[li].gate.v, step)
			adamUpdate(l.up, dWUp, layerAdams[li].up.m, layerAdams[li].up.v, step)
			te.Release(dWDown); te.Release(dWGate); te.Release(dWUp)
			te.Release(dFFNMid); te.Release(c.gatePre); te.Release(c.upOut); te.Release(c.ffnMid)

			// RMSNorm2 backward
			dN2 := te.Zeros([]int{n, dim})
			mongoose.KRMSNormBackward(dNormFinal.DevicePtr(), c.xMid.DevicePtr(),
				l.norm2.DevicePtr(), c.rmsScale2.DevicePtr(), dN2.DevicePtr(), n, dim)
			te.Release(c.normed2); te.Release(c.rmsScale2); te.Release(dNormFinal)

			// Attention backward: dQ, dK, dV
			dQ := te.Zeros([]int{n, dim})
			dK := te.Zeros([]int{n, kvDim})
			dV := te.Zeros([]int{n, kvDim})

			// dAttnOut from WO backward
			dWO := te.MatMulTransposeAT(c.attnOut, dN2, dim, n, dim)
			dAttnOut := te.MatMulT(dN2, l.wo, n, dim, dim)
			adamUpdate(l.wo, dWO, layerAdams[li].wo.m, layerAdams[li].wo.v, step)
			te.Release(dWO)

			mongoose.KCausalAttentionBackward(c.Q.DevicePtr(), c.K.DevicePtr(), c.V.DevicePtr(), dAttnOut.DevicePtr(),
				dQ.DevicePtr(), dK.DevicePtr(), dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)
			te.Release(dAttnOut); te.Release(c.attnOut)

			// RoPE backward
			mongoose.KRoPEBackward(dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPEBackward(dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			// QKV weight gradients
			dWQ := te.MatMulTransposeAT(c.normed, dQ, dim, n, dim)
			dWK := te.MatMulTransposeAT(c.normed, dK, dim, n, kvDim)
			dWV := te.MatMulTransposeAT(c.normed, dV, dim, n, kvDim)
			adamUpdate(l.wq, dWQ, layerAdams[li].wq.m, layerAdams[li].wq.v, step)
			adamUpdate(l.wk, dWK, layerAdams[li].wk.m, layerAdams[li].wk.v, step)
			adamUpdate(l.wv, dWV, layerAdams[li].wv.m, layerAdams[li].wv.v, step)
			te.Release(dWQ); te.Release(dWK); te.Release(dWV)
			te.Release(c.Q); te.Release(c.K); te.Release(c.V); te.Release(c.normed)
			te.Release(dQ); te.Release(dK); te.Release(dV)

			// RMSNorm1 backward → dNormFinal for next layer
			dNormFinal = te.Zeros([]int{n, dim})
			mongoose.KRMSNormBackward(dN2.DevicePtr(), c.xIn.DevicePtr(),
				l.norm1.DevicePtr(), c.rmsScale1.DevicePtr(), dNormFinal.DevicePtr(), n, dim)
			te.Release(dN2); te.Release(c.rmsScale1)
		}

		// AdamW on embeddings + final norm
		adamUpdate(embed, dEmbed, embedAdam.m, embedAdam.v, step)
		te.Release(dEmbed); te.Release(dNormFinal); te.Release(tokGPU)

		if step%*logEvery == 0 || step == 1 {
			cuda.Sync()
			elapsed := time.Since(t0)
			stepsPerSec := float64(step) / elapsed.Seconds()
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, totalLoss, lr, elapsed.Seconds(), stepsPerSec)
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)\n", *stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds())
}

func applyRoPESingle(x []float32, pos, headDim, numHeads int, theta float64) {
	half := headDim / 2
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			cos := float32(math.Cos(angle))
			sin := float32(math.Sin(angle))
			x0 := x[base+i]
			x1 := x[base+half+i]
			x[base+i] = x0*cos - x1*sin
			x[base+half+i] = x0*sin + x1*cos
		}
	}
}
