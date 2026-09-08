# SpeechPTQ

This repository contains the source code for the SpeechPTQ project, prepared for ICASSP.

## Repository Structure
- `src/`: Python source code for data processing, quantization, and evaluation.
- `scripts/`: Shell scripts for running ablations, benchmarking, and final experiments.

## Setup
Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Data and Models
The `.gitignore` excludes large files to keep the repository lightweight:
- `data/`: Place your datasets (e.g., LibriSpeech, AMI) here.
- `models/`: Place the downloaded model weights (e.g., Qwen3, Whisper) here.
- `logs/` and `results/`: Experiment outputs are generated and stored here automatically by the scripts.

## Note on Execution
Some scripts currently rely on specific pathing structures (`/workspace/SpeechPTQ/`). Be sure your environment mirrors this or update the scripts locally as needed.
