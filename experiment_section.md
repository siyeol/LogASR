# III. Experiment

## A. Models and hardware

We apply the proposed framework to two representative ASR architectures.
**Whisper-large-v3** [1] is a 1.55 B-parameter encoder–decoder Transformer whose decoder uses standard cross-attention over a fixed 1500-frame (30 s) encoder output.
**Qwen3-ASR-1.7B** [2] pairs a speech-specific encoder with a pre-trained LLM decoder, providing a setting where the decoder exhibits sharp channel outliers characteristic of text-domain models.
All experiments are conducted on a single NVIDIA RTX 3090 GPU (24 GB).
Whisper is loaded in `float16` and Qwen3-ASR in `bfloat16` via HuggingFace Transformers, with greedy decoding and language forced to English.

## B. Datasets and metrics

We evaluate on LibriSpeech [3] `test-clean` and `test-other` as the primary benchmarks.
For domain generalization we additionally report results on AMI [4] (meeting speech, IHM, 30 s stitched windows), Earnings-22 [5], VoxPopuli [6], GigaSpeech [7], and CommonVoice [8].
All audio is resampled to 16 kHz.

**Accuracy** is measured by word error rate (WER) computed with `jiwer` after standard text normalization.
**Efficiency** is assessed via four deployment-oriented metrics: (i) host-to-device (H2D) memory, measured as `torch.cuda.max_memory_allocated()` after model transfer; (ii) H2D loading time; (iii) throughput in utterances per second under a fixed 4 GB GPU memory budget; and (iv) real-time factor (RTF).

## C. Configurations and baselines

**Ours.** Following the analysis in Section II, we apply log-scale W4A4 quantization to encoder FFN layers and W4A8 to all remaining layers.
Weights use a single per-tensor scale; activations use a dynamic per-token scale.
For Qwen3-ASR, whose LLM decoder degrades under activation quantization, the decoder weights are loaded from a GPTQ [9] W4A16 checkpoint ($G{=}128$) while encoder quantization follows our method.
The KV-cache silence template is applied to Whisper by replacing zero-padded encoder positions with a pre-computed silence representation, sharing a single template across the batch.
All efficiency measurements use the packed `LogInt4` storage format (two 4-bit indices per `int8` byte, one `float16` per-tensor scale).

**Baselines.** We compare against:
(i) FP16 / BF16 full-precision inference;
(ii) W4A8 with group-128 uniform scaling, representing standard group-wise PTQ;
(iii) W4A8 weight-only grouping (activations kept at A16);
(iv) W4A4 with group-128 uniform scaling; and
(v) W4A4 weight-only grouping.
All grouped baselines use symmetric per-group `absmax` scaling.

---

**References**

[1] Radford et al., "Robust speech recognition via large-scale weak supervision," ICML 2023.
[2] Qwen Team, "Qwen3-ASR," 2025.
[3] Panayotov et al., "LibriSpeech: An ASR corpus based on public domain audio books," ICASSP 2015.
[4] Carletta et al., "The AMI meeting corpus," 2005.
[5] Del Rio et al., "Earnings-22: A practical benchmark for accents in the wild," InterSpeech 2022.
[6] Wang et al., "VoxPopuli: A large-scale multilingual speech corpus," ACL 2021.
[7] Chen et al., "GigaSpeech: An evolving, multi-domain ASR corpus," InterSpeech 2021.
[8] Ardila et al., "Common Voice: A massively-multilingual speech corpus," LREC 2020.
[9] Frantar et al., "GPTQ: Accurate post-training quantization for generative pre-trained transformers," ICLR 2023.
