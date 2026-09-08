# ICASSP GitHub Release Checklist

This document outlines the required steps to convert the active research repository into a clean, public-facing GitHub repository for ICASSP. 

Since the current directory is an active working environment, the recommended approach is to copy the source files to a **new, separate directory** and apply these changes there.

## Phase 1: Repository Skeleton & Copying
1. **Create the Release Directory:** Create a new folder (e.g., `SpeechPTQ_Release`) outside of your active working directory.
2. **Copy Source Code:** Copy *only* the following essential folders to the new directory:
   * `src/`
   * `scripts/`
3. **Move Stray Scripts:** 
   * Move `src/run_encoder_ablation.sh` to `scripts/`
   * Move `src/run_packing.sh` to `scripts/`
   * Move `src/sweep_w4a4.sh` to `scripts/`
4. **Isolate Paper Drafts:** 
   * If you wish to publish paper assets, create a `paper/` or `docs/` directory in the release repo and copy `.tex`, `.md`, `.pdf`, `.html`, and figure images (e.g., `comparison image.png`, `deftan-result.jpg`) into it. Otherwise, leave them behind completely.

## Phase 2: Pathing and Hardcode Fixes
The biggest blocker for external users is that scripts currently expect the repository to be located at `/workspace/SpeechPTQ/`. 

1. **Fix Shell Scripts:** 
   * Do a project-wide find-and-replace in `scripts/*.sh`.
   * Change all instances of `/workspace/SpeechPTQ/` to relative paths (e.g., `./`).
2. **Fix Python Scripts:** 
   * Do a project-wide find-and-replace in `src/*.py`.
   * Change `/workspace/SpeechPTQ/` to either relative paths, or dynamic paths utilizing `Path(__file__).resolve().parent.parent`.

## Phase 3: Setup Missing Configuration Files
Create the following files at the root of the new repository:

1. **Create `.gitignore`:**
   Ensure large files and cached elements are never committed.
   ```text
   # Python
   __pycache__/
   *.pyc
   .env
   
   # Data and Outputs
   data/
   logs/
   results/
   models/
   
   # OS generated files
   .DS_Store
   ```

2. **Create `requirements.txt` or `environment.yml`:**
   * Document all `pip` or `conda` dependencies needed to run the scripts (e.g., `torch`, `transformers`, `accelerate`).

3. **Create `README.md`:**
   * **Title & Paper Link:** Name of the project and link to the ICASSP paper.
   * **Installation:** Step-by-step instructions to create the python environment.
   * **Data Preparation:** Explain how a user can download `LibriSpeech` and put it in the `data/` folder.
   * **Model Weights:** Provide HuggingFace Hub links or bash commands to download the `.safetensors` model weights for Qwen3 and Whisper, explaining that they should be placed in `models/`.
   * **Usage:** Provide the exact bash commands (from your `scripts/` directory) required to reproduce the main results tables in the paper.

4. **Create `LICENSE`:**
   * Add a standard open-source license (e.g., MIT, Apache 2.0).

## Phase 4: Final Validation
Before running `git init` and pushing to GitHub:
1. Double-check that there are absolutely no `.safetensors` files, `data/` audio files, or `logs/` files hidden in the tree.
2. Run a basic script from the root directory to verify that the relative path changes work as expected.
