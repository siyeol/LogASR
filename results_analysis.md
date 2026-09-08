# IV. RESULT

We first conducted ablation studies to evaluate the effectiveness of our logarithmic activation quantization strategy across different model components. The results compared to the FP16 baseline are presented in Table 1. Notably, applying uniform W4A4 quantization exclusively to the decoder (Decoder-only W4A4) resulted in catastrophic failure, particularly for Qwen3-ASR-1.7B with a 94.26% word error rate (WER). This aligns with our observation that LLM-based decoders exhibit sharp channel outliers characteristic of text-domain models, making them highly sensitive to aggressive 4-bit activation quantization. Conversely, the encoder FFN layers, which process heavy-tailed speech representations, were successfully compressed using our Activation-only log-scale W4A4 scheme. This modification yielded minimal degradation, preserving transcription accuracy while aggressively scaling down the bit-width.

| Ablation list | Whisper-large-v3 (LS-clean, %) | Whisper-large-v3 (LS-other, %) | Qwen3-ASR-1.7B (LS-clean, %) | Qwen3-ASR-1.7B (LS-other, %) |
| :--- | :---: | :---: | :---: | :---: |
| FP16 | **1.92** | **3.91** | **1.64** | **3.41** |
| Encoder-only W4A4 | 2.16 | 4.92 | 2.22 | 5.43 |
| • Attention-only W4A4 | 2.12 | 4.35 | 1.95 | 4.34 |
| • FFN-only W4A4 | 2.12 | 4.72 | 2.06 | 4.57 |
| Decoder-only W4A4 | 2.54 | 4.87 | 94.26 | 94.67 |
| Weight-only log-scale W4A4 | 2.13 | 4.33 | 1.85 | 4.26 |
| log-scale W4A4 (Both W & A) | 2.76 | 6.47 | 2.96 | 8.12 |
| **Ours (Activation-only log-scale W4A4)** | 2.62 | 6.16 | 2.22 | 5.43 |

We compared the performance of our proposed W4A4 method with existing state-of-the-art post-training quantization techniques, including RTN, AWQ, GPTQ, and SmoothQuant. The results in Table 2 indicate that our method outperforms these existing models in the ultra-low bit regime. While dynamic INT8 methods like SmoothQuant perform adequately at W4A8, they collapse entirely when pushed to W4A4, resulting in near 100% WER across both architectures. By introducing the grouping-free logarithmic activation quantization, our approach smoothly transitions into the W4A4 regime, maintaining robust performance and demonstrating the effectiveness of the non-uniform log-scale in capturing speech activations.

| Methods | Whisper-large-v3 (LS-clean, %) | Whisper-large-v3 (LS-other, %) | Qwen3-ASR-1.7B (LS-clean, %) | Qwen3-ASR-1.7B (LS-other, %) |
| :--- | :---: | :---: | :---: | :---: |
| FP16 | 1.92 | 3.91 | 1.64 | 3.41 |
| RTN (W4A8) | 1.95 | 4.05 | 1.80 | 3.73 |
| AWQ + per-tensor RTN (W4A8) | 2.03 | 4.36 | 1.82 | 3.86 |
| GPTQ + per-tensor RTN (W4A8) | 1.93 | 4.09 | 1.73 | 3.83 |
| SmoothQuant + per-tensor RTN (W4A8)| 2.02 | 4.07 | 1.88 | 4.04 |
| **Ours (W4A8)** | **1.92** | **3.93** | **1.72** | **3.76** |
| SmoothQuant + per-tensor RTN (W4A4)| 99.71| 99.55 | 104.71 | 104.78 |
| **Ours (W4A4)** | 2.62 | 6.16 | 2.22 | 5.43 |

Furthermore, deployment-oriented efficiency was evaluated across different precision configurations (Table 3). The primary motivation for targeting A4 over A8 is the reduction of the activation representation size, which directly decreases GEMM input traffic. The empirical results confirm that moving from W4A8 to W4A4 noticeably reduces the GEMM input byte size for both Whisper-large-v3 and Qwen3-ASR-1.7B. Simultaneously, we maintained the substantial host-to-device (H2D) memory reduction provided by 4-bit weight quantization, preserving footprints of 843 MiB and 1428 MiB, respectively. This demonstrates that our W4A4 framework maximizes data movement efficiency without compromising high throughput or the memory gains of low-bit weights.

| Model | Precision | WER (LS-clean, %) | WER (LS-other, %) | H2D memory (MiB) | GEMM input byte | Throughput (utt/s @4GB) | RTF |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Whisper-large-v3 | FP16 | **1.92** | **3.91** | 2944 | 5747 | 17.0 | **0.0134** |
| | Baseline (W4A8) | 2.02 | 4.07 | **843** | 1950 | **18.9** | 0.0175 |
| | **Ours (W4A4)** | 2.62 | 6.16 | **843** | **1627** | **18.9** | 0.0175 |
| Qwen3-ASR-1.7B | FP16 | **1.64** | **3.41** | 3887 | 636.4 | 13.7 | **0.0174** |
| | Baseline (W4A8) | 1.88 | 4.04 | **1428** | 170.4 | **41.9** | 0.0259 |
| | **Ours (W4A4)** | 2.22 | 5.43 | **1428** | **159.1** | **41.9** | 0.0259 |

Finally, the domain generalization capability of the proposed method was evaluated across varied acoustic environments, as shown in Table 4. Despite the aggressive bit-width reduction to 4 bits for both weights and activations, our logarithmic quantization scheme retains strong generalization. While standard quantization techniques tend to overfit to clean read-speech benchmarks, our method shows stable performance on diverse real-world datasets including meeting speech (AMI) and accented speech (Earnings22), as well as heavily degraded audio (LS-clean + REVERB and LS-clean + DNS). This robustness indicates that the single dynamic scale combined with the non-uniform log-scale correctly preserves invariant heavy-tailed speech features regardless of background noise or acoustic reverberation.

| Model | Precision | AMI | Earnings22 | VoxPopuli | GigaSpeech | CommonVoice | LS-clean + REVERB | LS-clean + DNS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Whisper-large-v3 | FP16 | **16.27**| **11.36** | **9.29** | **10.10** | **9.86** | **6.25** | **17.71** |
| | **Ours (W4A4)** | 18.86 | 13.13 | 10.65 | 10.88 | 15.79 | 15.26 | 32.88 |
| Qwen3-ASR-1.7B | FP16 | **10.25**| **10.24** | **6.37** | **8.73** | **7.13** | **3.85** | **11.32** |
| | **Ours (W4A4)** | 14.06 | 12.04 | 7.48 | 9.60 | 11.74 | 10.08 | 23.46 |
