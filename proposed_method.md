# II. Proposed Method

We propose a unified lightweight framework that co-designs **bit-level compression** and **sequence-level optimization** for ASR models.
Section A introduces a grouping-free logarithmic post-training quantization scheme that enables W4A4 inference on the encoder FFN—the quantization bottleneck—while preserving recognition accuracy.
Section B presents a cross-attention KV-cache silence template that eliminates the redundant memory occupied by padded silence frames during autoregressive decoding.
The two techniques are orthogonal and compose without interference.

---

## A. Grouping-Free Log-Scale Post-Training Quantization

### A.1. Motivation: activation distribution in speech encoders

Standard uniform post-training quantization with per-tensor scaling fails at 4-bit precision (W4A4) because the symmetric grid wastes its limited representation budget on values that rarely occur.
To understand why, we profile the activation distributions of every linear-layer input across 2,620 utterances of LibriSpeech `test-clean`.
Two key observations emerge:

1. **Heavy-tailed, near-zero-concentrated mass.** Encoder FFN activations exhibit high excess kurtosis ($\kappa \gg 3$) and extreme dynamic range ($\log_2(\text{absmax}/p_{50}) \ge 8$). Over 99% of values fall below $A/7$ (where $A = \|x\|_\infty$), meaning a uniform 4-bit grid with step size $\Delta = A/7$ maps virtually the entire distribution into a single quantization bin.

2. **Dense channel-outlier tail.** Unlike text-domain LLMs, where a single channel dominates the per-tensor absmax, ASR encoder FFN inputs exhibit a *dense tail*—many channels are moderately elevated rather than one being sharply dominant. The peak ratio $r = \max_c |x_c| / \text{median}_c(|x_c|)$ is lower than in LLMs but spread across more channels. Consequently, per-tensor absmax is inflated by the aggregate of many channels, and the quantization error distributes broadly—precisely the regime where non-uniform compression is most beneficial.

These properties make encoder FFN layers both the dominant source of W4A4 degradation and the ideal target for logarithmic quantization: the distribution's mass near zero demands fine resolution at the origin, while the diffuse outlier structure makes group-wise handling unnecessary.

### A.2. Logarithmic quantization formulation

We define a symmetric 4-bit logarithmic codebook with $2 \times 7 + 1 = 15$ levels.
Given an input tensor $x$ (weight or activation), we first compute a scalar scale factor:

$$\alpha = \frac{A}{2^7 - 1} = \frac{A}{127}$$

where $A = \|x\|_\infty$ is the absmax of the tensor. This choice ensures the maximum value maps exactly to the grid boundary:

$$\log_2\!\left(1 + \frac{A}{\alpha}\right) = \log_2(128) = 7$$

**Quantization (forward):**

$$f(x) = \text{sign}(x) \cdot \log_2\!\left(1 + \frac{|x|}{\alpha}\right), \qquad y = \text{clip}\!\left(\text{round}(f(x)),\; -7,\; 7\right)$$

**Dequantization (reconstruction):**

$$\hat{x} = \text{sign}(y) \cdot \alpha \cdot \left(2^{|y|} - 1\right)$$

The reconstruction levels in absolute value are:

| $|y|$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **Uniform** | $0$ | $A/7$ | $2A/7$ | $3A/7$ | $4A/7$ | $5A/7$ | $6A/7$ | $A$ |
| **Log (ours)** | $0$ | $A/127$ | $3A/127$ | $7A/127$ | $15A/127$ | $31A/127$ | $63A/127$ | $A$ |

The first positive level is $A/127$ (log) versus $A/7$ (uniform), yielding $127/7 \approx 18\times$ finer resolution near the origin.
Uniform maps all $|x| < A/14$ to zero; the log grid's zero-bin boundary is $\approx A/254$, and its inter-level spacing follows a geometric progression ($\times 2$), keeping relative error bounded across the dynamic range.

**Empirical impact on encoder FFN activations.** On LibriSpeech `test-clean` (per-tensor $A$):
- $|x| < A/7$: **99.4%** of values → uniform collapses nearly all mass into one bin.
- $|x| < A/127$: **43%** of values → log preserves six additional levels in this region.
- Fraction mapped to zero: **~94%** (uniform) vs. **~36%** (log).
- Quantization MSE: uniform is **~11×** larger than log.

### A.3. Scaling strategy: group-free, per-tensor / per-token

For weights, $A = \max|W|$ is computed once per weight matrix (per-tensor absmax). A single `float16` scale $\alpha$ is stored per linear layer—yielding total weight scale storage on the order of **~1 KiB** for Whisper versus **~45 MiB** for a group-128 baseline.

For activations, $A = \max_{d}|x_{t,d}|$ is computed dynamically at each token position along the feature dimension (per-token, last-dim absmax). This yields a scalar scale per token at runtime, costing **~47 KiB** versus **~4.6 MiB** for group-128 on Whisper.

The group-free design eliminates:
- The storage of $\lceil K / G \rceil$ scale factors per weight row (typically $G = 128$).
- The gather/scatter overhead of group-wise dequantization during GEMM.
- The increased PCIe/memory-bus traffic from loading group scales.

### A.4. Hybrid precision allocation

Not all model components are equally sensitive to quantization. Through component-level evaluation (isolating encoder attention, encoder FFN, and decoder), we find:

- **Encoder FFN** is the bottleneck but tolerates log W4A4 well (FFN collapse rate drops from ~93% under uniform to ~34% under log).
- **Encoder self-attention** is moderately sensitive; W4A8 suffices.
- **Decoder** varies by architecture: Whisper's standard decoder tolerates W4A8, but Qwen3-ASR's LLM decoder collapses under any activation quantization below 16-bit.

Our final configuration: **Encoder FFN → W4A4 (log, group-free)**, **all other layers → W4A8** (log weights, dynamic per-token 8-bit uniform activations). For Qwen3-ASR, the LLM decoder uses GPTQ W4A16 ($G = 128$).

### A.5. Hardware-accelerated INT8 tensor-core execution

The log codebook is non-affine, so standard INT4 matrix-multiply-accumulate on the raw indices would produce incorrect results. However, every reconstruction value $z = \text{sign}(y) \cdot (2^{|y|} - 1)$ fits within signed INT8:

$$z \in \{-127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31, 63, 127\}$$

This enables a three-stage execution pipeline:

1. **Activation quantization** (Triton kernel): A fused elementwise kernel reads the FP16 activation tensor, computes the log transform, applies rounding and clamping, and outputs INT8 reconstruction values via a 15-entry lookup table (LUT). For per-token scaling, the kernel loads one $\alpha$ per row.

2. **INT8 GEMM** (`torch._int_mm`): The INT8 activation matrix $Z_A$ is multiplied by the pre-stored INT8 weight matrix $W_{\text{INT8}}^{\top}$ using cuBLAS INT8 tensor cores (Ampere+), producing an INT32 accumulator.

3. **Fused rescaling** (Triton kernel): A second kernel multiplies the INT32 accumulator by $\alpha_a \cdot \alpha_w$, adds FP16 bias, and writes the FP16 output—all in a single pass.

**Weight storage.** Weights are packed as 4-bit unsigned nibbles (two per `uint8` byte, shifted by $+7$ to $[0, 14]$), achieving full $4\times$ compression. At GEMM time, the pre-computed INT8 LUT reconstruction matrix ($[K, N]$, `int8`) is used directly, avoiding runtime dequantization.

---

## B. Cross-Attention KV-Cache Silence Template

### B.1. The padding overhead problem

Whisper processes audio in fixed 30-second windows, producing $T = 1500$ encoder frames regardless of actual speech duration. During autoregressive decoding, the decoder cross-attention computes keys and values over all $T$ frames, storing:

$$\text{KV memory} = L \times 2 \times T \times d_{\text{model}} \times b_{\text{dtype}}$$

where $L = 32$ layers, $d_{\text{model}} = 1280$, $b_{\text{dtype}} = 2$ bytes (FP16), giving **~234 MiB per utterance** for the full 1500 frames. On LibriSpeech `test-clean`, the mean utterance duration is 7.4 s (~370 frames), meaning **over 75% of the KV cache stores representations of zero-padded silence**.

### B.2. Pre-computed silence template

We eliminate this waste by pre-computing a canonical silence template. Offline, 30 seconds of pure silence is processed through the encoder once, yielding a position-aware template $\mathbf{T} \in \mathbb{R}^{1500 \times d_{\text{model}}}$.

At inference time, a lightweight forward hook is registered on the encoder output. For each utterance $i$ with $n_i$ speech frames:

$$\text{encoder\_hidden}[i,\; n_i:] \leftarrow \mathbf{T}[n_i:]$$

where $n_i = \lfloor \text{n\_samples}_i / 320 \rfloor$ (320 samples = 1 encoder frame at 16 kHz with hop 160 and conv stride 2).

Because the template entries are identical across all utterances in a batch, the decoder cross-attention keys and values for positions $\ge n_i$ can be shared. Only the $n_i$ speech-bearing entries require per-utterance storage, reducing effective KV memory to:

$$\text{KV memory (ours)} = L \times 2 \times n_{\text{speech}} \times d_{\text{model}} \times b_{\text{dtype}} + \underbrace{L \times 2 \times T \times d_{\text{model}} \times b_{\text{dtype}}}_{\text{one shared template}}$$

For a batch of $B$ utterances with mean speech length $\bar{n}$, the saving over the baseline ($B \times T$ stored entries) is:

$$\text{Saving} = 1 - \frac{B \cdot \bar{n} + T}{B \cdot T}$$

### B.3. Variable-length encoder packing

As a complementary technique, we patch the Whisper encoder to accept variable-length mel-spectrogram inputs rather than the rigid 3000-frame (30 s) padding. The positional embedding table is dynamically sliced to the actual input length $t$:

$$\mathbf{h} = \text{ConvSubsample}(\mathbf{x}_{\text{mel}}) + \mathbf{E}_{\text{pos}}[:t]$$

This removes the padding entirely, eliminating both the encoder computation and the KV storage for silence frames. The two stride-2 1D convolutions require an even number of input mel frames; odd inputs are right-padded by one frame before convolution.

### B.4. Token-aligned silence collapse

For Qwen3-ASR, whose audio tokenizer operates at 12.5 Hz (80 ms per token via $8\times$ mel downsampling), we apply energy-based silence detection directly on the raw waveform with frame alignment to the tokenization rate.

The procedure:
1. **Frame-level RMS energy** is computed over non-overlapping 80 ms windows (1,280 samples at 16 kHz):

$$\text{RMS}_k = \sqrt{\frac{1}{L}\sum_{m=0}^{L-1} x_{k,m}^2}$$

2. **Adaptive thresholding** determines speech/silence per frame:

$$\text{threshold} = \max\!\left(\text{peak} \cdot 10^{\text{rel\_dB}/20},\; 10^{-4}\right)$$

where $\text{peak} = \max_k \text{RMS}_k$ and $\text{rel\_dB} = -35$ dB. Frame $k$ is classified as speech if $\text{RMS}_k \ge \text{threshold}$.

3. **Run-length collapse** replaces each consecutive silence run of $\ge 2$ frames with a single hold frame (80 ms), preserving prosodic boundary cues while removing sustained pauses. Silence runs shorter than 2 frames are kept intact to avoid fragmenting brief inter-word gaps.

The collapsed waveform is then passed to Qwen3-ASR's audio processor, producing a shorter token sequence that directly reduces encoder computation and KV-cache size.

---

## C. Framework Summary

The complete framework operates in three stages with no retraining or calibration data:

1. **Weight quantization** (offline): Log-codebook W4 with per-tensor scale → packed 4-bit nibble storage.
2. **Activation quantization** (runtime): Dynamic per-token log A4 on encoder FFN, uniform A8 elsewhere → fused Triton kernel to INT8 → tensor-core GEMM.
3. **Silence optimization** (runtime): Template replacement (Whisper) or waveform collapse (Qwen3-ASR) → reduced sequence length and KV-cache footprint.

All three stages require only a single forward pass through the original model—no gradient computation, no calibration set, and no architecture modification.
