#!/bin/bash
# Monitor baseline training progress
# Paths resolve relative to the repo root (parent of this script's dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG="$PROJECT_ROOT/outputs/baseline_training.log"
CHECKPOINT_DIR="$PROJECT_ROOT/../vit_swinunetr_segmentation/checkpoints"

while true; do
    # Check if process is still running
    if ! pgrep -f "scripts/train.py" > /dev/null; then
        echo "[$(date)] Training process has stopped"
        # Check if it finished successfully
        if grep -q "Training complete" "$LOG" 2>/dev/null; then
            echo "[$(date)] Training completed successfully!"
            BEST_LINE=$(grep "Training complete" "$LOG" | tail -1)
            echo "$BEST_LINE"
        else
            echo "[$(date)] Training may have crashed. Last lines:"
            tail -c 2000 "$LOG" | strings | grep -E "Epoch|Validation|Error|memory" | tail -5
        fi
        break
    fi
    
    # Get latest epoch and validation info
    LATEST_EPOCH=$(grep -a "INFO - Epoch" "$LOG" | tail -1)
    LATEST_VAL=$(grep -a "Validation" "$LOG" | tail -1)
    GPU_MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    
    echo "[$(date)] $LATEST_EPOCH | GPU: $GPU_MEM"
    if [ -n "$LATEST_VAL" ]; then
        echo "  Last validation: $LATEST_VAL"
    fi
    
    sleep 600  # Check every 10 minutes
done
