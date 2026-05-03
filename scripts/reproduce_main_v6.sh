#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jaemo/AVVP_stage12_clean"
PYBIN="/home/jaemo/miniconda3/envs/av2a_fresh/bin/python"
OUT_DIR="${1:-${ROOT}/results/main_v6_lam0p3_k4_e1_esc50_noq_mindur2}"

"${PYBIN}" "${ROOT}/run_llp_stage12.py" \
  --out-dir "${OUT_DIR}" \
  --lambda-base 0.3 \
  --kappa 4 \
  --eta 1 \
  --rho-min 0.1 \
  --rho-max 1.0 \
  --stage2-prior-mode full \
  --stage2-active-set all \
  --video-prior-no-reliability \
  --mean-source external \
  --audio-mean-path /mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clap_HTSAT-tiny_audio_esc50_N1600.npy \
  --visual-mean-path /mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clip_ViT-L-14_image_mscoco_train_N118287.npy \
  --score-mode adaptive_k \
  --score-k0 16 \
  --score-t-min 0.25 \
  --score-t-max 1.25 \
  --score-thr 0.75 \
  --pred-min-duration 2 \
  --fista-iters 200 \
  --device cuda \
  --no-details
