# Sensor2Sensor — project notes for Claude

## Default conda environment

Use **`selfocc`** (`/home/satya/conda_envs/selfocc`) for all work in this repo unless told otherwise.

Activate via:

```bash
source /home/satya/anaconda3/etc/profile.d/conda.sh
conda activate /home/satya/conda_envs/selfocc
```

Key versions: Python 3.8.16, torch 2.0.0+cu118, numpy 1.24.4, pandas 2.0.3. Install any missing extras (pyarrow, diffusers, huggingface_hub, safetensors, gsutil, etc.) into this env via pip.
