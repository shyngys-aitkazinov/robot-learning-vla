#!/usr/bin/env bash
# Periodic status snapshot for a running Eval3 SmolVLA training job.
#
# Polls every N seconds (default 60), prints a single status line each tick:
#   step / total / pct / loss / grad / lr / step-rate / ETA
#   GPU VRAM used / total / util%
#   training-process CPU% + RSS
#
# Designed to be backgrounded:
#   ./tools/eval3_train_monitor.sh outputs/train/logs/<run>.log \
#     > outputs/train/logs/<run>_monitor.log 2>&1 &
#
# Args:
#   $1 (required) — training log file (the one being tee'd to by the run)
#   $2 (optional) — poll interval in seconds, default 60
#   $3 (optional) — pgrep pattern for the train process, default 'train_eval3_smolvla.*--steps='

set -u

LOG="${1:?usage: eval3_train_monitor.sh <training_log> [poll_seconds] [pgrep_pattern]}"
INTERVAL="${2:-60}"
PIDPAT="${3:-train_eval3_smolvla.*--steps=}"

last_step=""
last_ts_epoch=""

while true; do
  ts=$(date '+%Y-%m-%dT%H:%M:%S')
  ts_epoch=$(date +%s)

  # Pull the most recent INFO step line (lerobot logs step:N smpl:M ep:E epch:F loss:L grdn:G lr:R)
  step_line=$(grep -E "INFO.*step:[0-9]+" "$LOG" 2>/dev/null | tail -1)

  if [[ -z "$step_line" ]]; then
    printf "[%s]  (no training step lines yet — still in stats merge or pre-init)\n" "$ts"
  else
    step=$(echo "$step_line" | grep -oE "step:[0-9]+" | head -1 | cut -d: -f2)
    loss=$(echo "$step_line" | grep -oE "loss:[0-9.eE+-]+" | head -1 | cut -d: -f2)
    grdn=$(echo "$step_line" | grep -oE "grdn:[0-9.eE+-]+" | head -1 | cut -d: -f2)
    lr=$(echo "$step_line" | grep -oE "lr:[0-9.eE+-]+" | head -1 | cut -d: -f2)
    epch=$(echo "$step_line" | grep -oE "epch:[0-9.]+" | head -1 | cut -d: -f2)

    # Step rate + ETA using delta against the previous tick
    rate_str="-"
    eta_str="-"
    if [[ -n "$last_step" && -n "$last_ts_epoch" && "$step" != "$last_step" ]]; then
      ds=$(( step - last_step ))
      dt=$(( ts_epoch - last_ts_epoch ))
      if (( dt > 0 && ds > 0 )); then
        rate=$(awk "BEGIN{printf \"%.2f\", $ds / $dt}")
        # Pull total steps from the cmdline of the train process
        tot=$(pgrep -af "$PIDPAT" | head -1 | grep -oE "\-\-steps=[0-9]+" | head -1 | cut -d= -f2)
        if [[ -n "$tot" && "$rate" != "0.00" ]]; then
          remaining=$(( tot - step ))
          eta_s=$(awk "BEGIN{printf \"%d\", $remaining / $rate}")
          eta_h=$(( eta_s / 3600 ))
          eta_m=$(( (eta_s % 3600) / 60 ))
          eta_str="${eta_h}h${eta_m}m"
        fi
        rate_str="${rate} step/s"
      fi
    fi
    last_step="$step"
    last_ts_epoch="$ts_epoch"

    # GPU
    gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    vram_used=$(echo "$gpu" | cut -d, -f1)
    vram_tot=$(echo "$gpu" | cut -d, -f2)
    gpu_util=$(echo "$gpu" | cut -d, -f3)

    # Training process CPU + RSS
    pid=$(pgrep -f "$PIDPAT" | head -1)
    cpu_str="-"
    rss_str="-"
    if [[ -n "$pid" ]]; then
      cpu_str=$(ps -p "$pid" -o %cpu= 2>/dev/null | xargs)
      rss_kb=$(ps -p "$pid" -o rss= 2>/dev/null | xargs)
      if [[ -n "$rss_kb" ]]; then
        rss_str=$(awk "BEGIN{printf \"%.1f GB\", $rss_kb / 1024 / 1024}")
      fi
    fi

    printf "[%s]  step:%s  loss:%s  grd:%s  lr:%s  ep:%s  %s  ETA:%s  | VRAM %s/%s MB (util %s%%)  | CPU %s%% RSS %s\n" \
      "$ts" "$step" "$loss" "$grdn" "$lr" "$epch" "$rate_str" "$eta_str" \
      "$vram_used" "$vram_tot" "$gpu_util" "$cpu_str" "$rss_str"
  fi

  sleep "$INTERVAL"
done
