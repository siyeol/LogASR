# Log_ASR

Reference implementation of **Log-ASR: Logarithmic Activation Quantization and
Silence Frame Packaging for 4-Bit Automatic Speech Recognition** for:

- `openai/whisper-large-v3`
- `Qwen/Qwen3-ASR-1.7B`

The repository contains only the proposed Log-ASR quantizer and Silence Frame
Packaging (SFP), plus evaluation scripts. Comparison baselines and plotting code
are intentionally excluded.

## Quantization configuration

### Log-ASR activation quantization

For every activation token, one scale is shared across all channels:

```text
M     = 2^(b-1) - 1 = 7
mu    = 2^M - 1 = 127
A_t   = max_c |x[t,c]|
alpha = A_t / 127
q     = clip(round(sign(x) * log2(1 + |x| / alpha)), -7, 7)
x_hat = sign(q) * alpha * (2^|q| - 1)
```

This is a **grouping-free, per-token** activation quantizer. Weight groups do
not partition the activation channels.

### Model-specific precision

| Model component | Weight | Activation | Weight group |
|---|---:|---:|---:|
| Whisper encoder + decoder | uniform W4 | Log-ASR A4 | 64 |
| Qwen3-ASR audio encoder | uniform W4 | Log-ASR A4 | 64 |
| Qwen3-ASR text decoder | GPTQ W4 | BF16 (A16) | 128 |

Runtime Log-A4 uses PyTorch quantize-dequantize simulation and is not an
optimized INT4 activation kernel. Whisper W4 weights and the Qwen GPTQ decoder
are stored as packed `compressed-tensors` checkpoints; the reference loader
decompresses them before applying the Log-A4 runtime recipe.

## Silence Frame Packaging

SFP divides 16 kHz audio into non-overlapping 80 ms frames. A frame is silent
when

```text
RMS(frame) < max(utterance_peak_RMS * 10^(-35/20), 1e-4).
```

For each silence run containing at least two frames, SFP retains the **first
frame** and removes the remainder. Speech frames and short pauses remain
unchanged.

## Checkpoints

Checkpoints are intentionally excluded from Git because of their size. Generate
them locally with the commands below. Each output directory contains
`recipe_w4a4.json` and `recipe_w4a4_sfp.json`; SFP has no learned parameters,
so the two recipes share one weight checkpoint.

## Installation

Python 3.10+ and CUDA-capable PyTorch are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Whisper

Create the packed W4 (G=64) checkpoint:

```bash
python export_whisper_checkpoint.py \
  --output checkpoints/whisper-large-v3-log-asr-w4a4-g64
```

Transcribe:

```bash
python transcribe.py audio.wav \
  --model-id checkpoints/whisper-large-v3-log-asr-w4a4-g64 \
  --mode w4a4 --silence-pack
```

Evaluate LibriSpeech:

```bash
python evaluate_librispeech.py \
  --data-root /path/to/LibriSpeech \
  --split test-clean \
  --mode w4a4 \
  --model-id checkpoints/whisper-large-v3-log-asr-w4a4-g64 \
  --silence-pack \
  --batch-size 4 \
  --output results/whisper_w4a4_sfp_test-clean.json
```

## Qwen3-ASR

### 1. Calibrate the decoder

The paper uses GPTQ W4A16 with 128 randomly selected LibriSpeech samples,
maximum sequence length 2048, and weight group size 128:

```bash
python calibrate_qwen_gptq.py \
  --data-root /path/to/LibriSpeech \
  --split dev-clean \
  --samples 128 \
  --max-seq-length 2048 \
  --output checkpoints/qwen3-asr-1.7b-log-asr-w4a4-decoder-gptq-w4a16-g128
```

The sequential calibration script excludes `audio_tower` and `lm_head`, saves the complete
ASR checkpoint, reloads it, and rejects a checkpoint that produces an empty or
`ModelError` smoke-test transcription. Calibration is a one-time, GPU-intensive
operation.

### 2. Transcribe

```bash
python transcribe_qwen.py audio.wav \
  --mode w4a4 \
  --decoder-checkpoint checkpoints/qwen3-asr-1.7b-log-asr-w4a4-decoder-gptq-w4a16-g128 \
  --silence-pack
```

### 3. Evaluate LibriSpeech

```bash
python evaluate_qwen_librispeech.py \
  --data-root /path/to/LibriSpeech \
  --split test-clean \
  --mode w4a4 \
  --decoder-checkpoint checkpoints/qwen3-asr-1.7b-log-asr-w4a4-decoder-gptq-w4a16-g128 \
  --silence-pack \
  --batch-size 1 \
  --output results/qwen_w4a4_sfp_test-clean.json
```

W4A4 evaluation refuses checkpoints without GPTQ/compressed-tensors metadata so
an unquantized or partially written decoder cannot be mislabeled as W4A16. SFP
has no learned parameters, so each model uses one weight checkpoint plus the
`recipe_w4a4.json` and `recipe_w4a4_sfp.json` runtime recipes.

## Metrics

Evaluation JSON files report corpus WER, RTF, audio duration before/after SFP,
sequence reduction, peak CUDA memory, batch size, and explicit encoder/decoder
precision. Whisper reports encoder frames; Qwen3-ASR reports audio tokens. RTF
excludes model loading and one-time weight conversion.

## Tests

```bash
pytest -q
```
