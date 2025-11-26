# Fara-7B Setup on RTX 5090 (Blackwell Architecture)

This document describes the complete setup process for running Microsoft's Fara-7B agentic model on an NVIDIA RTX 5090 GPU.

## Table of Contents
1. [Setup Overview](#setup-overview)
2. [Installation Steps](#installation-steps)
3. [Problems Encountered & Solutions](#problems-encountered--solutions)
4. [Running the Model](#running-the-model)

---

## Setup Overview

**Hardware**: NVIDIA RTX 5090 (32GB VRAM, Blackwell architecture sm_120)
**Model**: Microsoft Fara-7B (~15GB)
**Framework**: vLLM 0.11.2 with PyTorch 2.9.0+cu128

### Key Components
- **Conda Environment**: `fara` (Python 3.12)
- **Model Location**: `~/.cache/huggingface/hub/models--microsoft--Fara-7B/snapshots/main/`

---

## Installation Steps

### 1. Create Conda Environment
```bash
conda create -n fara python=3.12 -y
conda activate fara
```

### 2. Install Fara Package
```bash
cd /home/ye/ml-experiments/fara
pip install -e .
```

### 3. Install PyTorch with Blackwell Support
The RTX 5090 uses CUDA compute capability sm_120 (Blackwell), which requires PyTorch nightly builds or recent releases with CUDA 12.8 support.

```bash
# Install vLLM pre-release (automatically pulls compatible PyTorch)
pip install vllm --pre

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
PyTorch: 2.9.0+cu128
CUDA: True
Device: NVIDIA GeForce RTX 5090
```

### 4. Install Transformers (Compatible Version)
```bash
pip install transformers==4.56.0 tokenizers>=0.21.1
```

### 5. Install Playwright Browsers
```bash
playwright install
```

### 6. Download Fara-7B Model
```bash
# Using huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='microsoft/Fara-7B',
    local_dir='~/.cache/huggingface/hub/models--microsoft--Fara-7B/snapshots/main',
    local_dir_use_symlinks=False
)
"
```

Or use the provided script:
```bash
python scripts/download_model.py
```

---

## Problems Encountered & Solutions

### Problem 1: CUDA Architecture Not Supported
**Error**:
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation
```

**Cause**: Standard PyTorch builds don't include sm_120 (Blackwell) support.

**Solution**: Install vLLM pre-release which pulls PyTorch 2.9.0+cu128 with Blackwell support:
```bash
pip install vllm --pre
```

### Problem 2: CUDA Out of Memory During Video Profiling
**Error**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.38 GiB
```

**Cause**: vLLM's video encoder profiling tries to allocate too much memory during initialization.

**Solution**: Use `--enforce-eager` flag and lower `--gpu-memory-utilization`:
```bash
--enforce-eager --gpu-memory-utilization 0.85
```

### Problem 3: Transformers Version Incompatibility
**Error**:
```
AttributeError: 'dict' object has no attribute 'model_type'
```

**Cause**: transformers 4.57.x has a tokenizer loading bug with vLLM.

**Solution**: Downgrade to transformers 4.56.0:
```bash
pip install transformers==4.56.0
```

### Problem 4: Model Name Mismatch
**Error**:
```
The model `Fara-7B` does not exist
```

**Cause**: The default endpoint config uses model name "Fara-7B" but vLLM serves it as "gpt-4o-mini-2024-07-18".

**Solution**: Create endpoint config with matching model name:
```json
{
    "model": "gpt-4o-mini-2024-07-18",
    "base_url": "http://localhost:5000/v1",
    "api_key": "not-needed"
}
```

### Problem 5: Image Limit Too Low
**Error**:
```
At most 1 image(s) may be provided in one prompt
```

**Cause**: `--limit-mm-per-prompt` set too low for multi-turn conversations with screenshots.

**Solution**: Increase image limit:
```bash
--limit-mm-per-prompt '{"image": 10}'
```

### Problem 6: Context Length Too Short
**Error**:
```
The decoder prompt (length 4835) is longer than the maximum model length of 4096
```

**Cause**: Images consume many tokens; 4096 is insufficient for vision-language tasks.

**Solution**: Increase max model length:
```bash
--max-model-len 16384
```

---

## Running the Model

### Step 1: Start the vLLM Server

```bash
conda activate fara

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 5000 \
  --model ~/.cache/huggingface/hub/models--microsoft--Fara-7B/snapshots/main \
  --served-model-name gpt-4o-mini-2024-07-18 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 10}'
```

**Server Parameters Explained**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--gpu-memory-utilization` | 0.85 | Use 85% of GPU memory (~27GB of 32GB) |
| `--enforce-eager` | - | Disable CUDA graphs to reduce memory spikes |
| `--max-model-len` | 16384 | Maximum context length (16k tokens) |
| `--limit-mm-per-prompt` | `{"image": 10}` | Allow up to 10 images per prompt |
| `--served-model-name` | gpt-4o-mini-2024-07-18 | Model name for API compatibility |

### Step 2: Run the Fara Agent

```bash
python test_fara_agent.py \
  --task "Search for 'your query' on Bing" \
  --start_page "https://www.bing.com/" \
  --max_rounds 10 \
  --save_screenshots \
  --downloads_folder /tmp/fara_output \
  --endpoint_config endpoint_configs/local_vllm.json
```

**Agent Parameters**:
| Parameter | Description |
|-----------|-------------|
| `--task` | Natural language task description |
| `--start_page` | Initial URL to navigate to |
| `--max_rounds` | Maximum agent action iterations |
| `--save_screenshots` | Save screenshots of each step |
| `--downloads_folder` | Output directory for screenshots |
| `--headful` | (Optional) Show browser GUI |
| `--endpoint_config` | Path to model endpoint config |

### Step 3: Verify Server Health

```bash
curl http://localhost:5000/health
# Should return HTTP 200
```

---

## Quick Start Commands

```bash
# Terminal 1: Start server
conda activate fara
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 5000 \
  --model ~/.cache/huggingface/hub/models--microsoft--Fara-7B/snapshots/main \
  --served-model-name gpt-4o-mini-2024-07-18 \
  --gpu-memory-utilization 0.85 --trust-remote-code --enforce-eager \
  --max-model-len 16384 --limit-mm-per-prompt '{"image": 10}'

# Terminal 2: Run agent
conda activate fara
python test_fara_agent.py \
  --task "Go to google.com and search for 'weather today'" \
  --start_page "https://www.google.com/" \
  --max_rounds 5 \
  --endpoint_config endpoint_configs/local_vllm.json
```

---

## File Locations

| Item | Path |
|------|------|
| Model weights | `~/.cache/huggingface/hub/models--microsoft--Fara-7B/snapshots/main/` |
| Endpoint config | `endpoint_configs/local_vllm.json` |
| Test script | `test_fara_agent.py` |

---

## Memory Usage

With the recommended settings on RTX 5090 (32GB):
- Model weights: ~15.6 GiB
- KV cache: ~8.4 GiB (157,120 tokens)
- Maximum concurrency: ~9.6x for 16k context

---

*Document created: November 2025*
