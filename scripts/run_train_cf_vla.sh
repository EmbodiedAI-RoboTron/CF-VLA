#!/bin/bash

set -x

CONFIG_NAME=$1
shift || true
OTHER_ARGS=("$@")

if [ -z "$CONFIG_NAME" ]; then
  echo "Usage: $0 <config_yaml_path> [other_args]"
  exit 1
fi

if [ ! -f "$CONFIG_NAME" ]; then
  echo "Config file not found: $CONFIG_NAME"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CONFIG_NAME")"

EXPNAME=$(basename "$CONFIG_NAME" | sed 's/\.yaml$//' | sed 's/\.yml$//')
PARENT_DIR=$(basename "$(dirname "$CONFIG_NAME")")

checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-"${ROOT_DIR}/outputs/checkpoints"}
assets_base_dir=${ASSETS_BASE_DIR:-"${ROOT_DIR}/assets"}

export PYTHON=${PYTHON:-python}
export TORCHRUN=${TORCHRUN:-torchrun}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-~/.cache/hf_datasets}
export HF_HOME=${HF_HOME:-~/.cache/hf_home}
export TMPDIR=${TMPDIR:-/tmp}
export MODEL_ZOO=${MODEL_ZOO:-"${ROOT_DIR}/checkpoints"}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"${MODEL_ZOO}"}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-"${ROOT_DIR}/data/lerobot"}

export SWANLAB_PROJECT=${PARENT_DIR}/${EXPNAME}

EXP_DIR="$checkpoint_base_dir/$PARENT_DIR/$EXPNAME"
mkdir -p "$EXP_DIR"

cp -r "$ROOT_DIR/src" "$EXP_DIR/"
cp -r "$ROOT_DIR/scripts" "$EXP_DIR/"
cp "$CONFIG_NAME" "$EXP_DIR/$PARENT_DIR.yaml"
CONFIG_NAME="$PARENT_DIR.yaml"

export PYTHONPATH="$EXP_DIR/src":$PYTHONPATH
cd "$EXP_DIR" || exit 1

GPUS_NUM=1
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "Checking GPU availability..."
    while true; do
        CLEAN="${CUDA_VISIBLE_DEVICES// /}"
        if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "$CLEAN" ]; then
            gpu_usage=$(nvidia-smi -i "$CLEAN" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        else
            gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        fi
        all_free=true
        while IFS= read -r usage; do
            if [ "$usage" -gt 10 ]; then
                all_free=false
                break
            fi
        done <<< "$gpu_usage"

        if [ "$all_free" = true ]; then
            echo "All GPUs are idle. Start training..."
            break
        else
            echo "GPU is busy. Recheck in 30 seconds..."
            sleep 30
        fi
    done

    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      CLEAN="${CUDA_VISIBLE_DEVICES// /}"
      if [ -n "$CLEAN" ]; then
        GPUS_NUM=$(awk -F',' '{print NF}' <<<"$CLEAN")
      fi
    else
      GPUS_NUM=$(echo "$gpu_usage" | wc -l)
    fi
fi

SWANLAB_API_KEY=${SWANLAB_API_KEY:-} "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$GPUS_NUM"   scripts/train_pytorch.py "$CONFIG_NAME"   --exp-name="$EXPNAME" --overwrite   --checkpoint-base-dir="$EXP_DIR"   --assets-base-dir="$assets_base_dir"   "${OTHER_ARGS[@]}"   2>&1 | tee -a log.txt
