# AVVP Stage1/2 Clean Implementation

이 폴더는 `/home/jaemo/AVVP` 안의 기존 실험 코드를 참고해서,
LLP-25 기반 **stage1 independent decomposition** 과
**v6-main Method** 의 cross-modal prior 기반
hint-weighted sparse decomposition을 깔끔하게 다시 구현한 버전이다.

현재 범위:
- stage1: modality별 centered non-negative Lasso decomposition
- stage2: reliability-weighted sparse confidence `g(t,c)` 에서 segment prior `h(t,c)` 와 video prior `pi(c)` 를 대칭적으로 구성
- stage3: `lambda_c(t) = lambda_base * clip(exp(-eta * H(t,c)), rho_min, rho_max)` 로 weighted Lasso 재실행
- stage4: v5.2 sparsity-aware fixed-threshold AVVP scoring + min-duration decoding + detail report

v6-main default result on LLP test (1108 valid clips, AV2A official metric):

| Method | A_seg | A_evt | V_seg | V_evt | AV_seg | AV_evt | Type@seg | Type@evt | Event@seg | Event@evt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage2 + K-temp (no q, min-dur>=2) | 41.31 | 30.36 | 57.15 | 50.25 | 47.20 | 33.89 | 48.55 | 38.17 | 41.33 | 31.16 |

이번 구현에서 의도적으로 제외한 것:
- explicit absence / active rejection main path
- weak-label gating
- 기존 `/home/jaemo/AVVP/cross_modal_hint` 의 ad-hoc path / legacy assets

## Feature / vocab source

- audio segment: `/mnt/hdd4tb/jaemo/data/LLP/cached_avvp/audio/ClipClap/<video>/0_10.pt`
- visual segment: `/mnt/hdd4tb/jaemo/data/LLP/cached_avvp/image/ClipClap/<video>/0_10.pt`
- global audio: `/mnt/hdd4tb/jaemo/data/LLP/cached_avvp/global_audio/ClipClap/<video>/0_10.pt`
- vocab: `/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/vocabs/v25_{clip,clap}.npy`

`v25_clip.npy`, `v25_clap.npy` 는 prompt-encoded LLP-25 prototype이다:
- visual prompt: `a photo of {class}`
- audio prompt: `a sound of {class}`

## Centering

segment-level:

```python
z_n = normalize(z)
z_tilde = normalize(z_n - mu_seg)
```

video-level:

```python
z_video_n = normalize(z_video)
z_video_tilde = normalize(z_video_n - mu_video)
```

prototype:

```python
C_n = normalize(C)
C_tilde = normalize(C_n - mean(C_n))
```

여기서 `mu_seg` 와 `mu_video` 는 modality별 dataset mean 이다.
v5 main path에서는 video-level sparse decomposition을 사용하지 않지만,
cached global feature shape / legacy comparison을 위해 bundle에는 유지한다.

실험용으로 `run_llp_stage12.py` 는 mean source를 둘 다 지원한다.
- `--mean-source llp`: LLP test feature mean
- `--mean-source external`: precomputed backbone reference mean
  - visual default: `/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clip_ViT-L-14_image_mscoco_train_N118287.npy`
  - audio default: `/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clap_HTSAT-tiny_audio_esc50_N1600.npy`

## Weighted hint

v5 main formulation은 explicit absence term을 쓰지 않는다.
먼저 각 segment-class pair에 대해 reliability-weighted sparse confidence를 정의한다.
Segment-level local support와 video-level prior는 모두 이 동일한 confidence에서 파생되며,
차이는 temporal scope뿐이다.

```python
s_m[t, c] = w_m[t, c] / (max_j w_m[t, j] + eps)
q_m[t] = max(0, cos(z_tilde_m[t], normalize(C_tilde_m @ w_m[t])))
g_m[t, c] = s_m[t, c] * q_m[t]
h_m[t, c] = g_m[t, c]
pi_m[c] = max_t s_m[t, c]  # v6 default; --video-prior-use-reliability restores max_t g_m[t,c]
H_source_to_target[t, c] = pi_source[c] * (1 + kappa * g_source[t, c])
lambda_target[t, c] = lambda_base * clip(exp(-eta * H_source_to_target[t, c]), rho_min, rho_max)
```

audio reweight에는 visual evidence를, visual reweight에는 audio evidence를 넣는다.
`--stage2-prior-mode video` 또는 `--video-prior-only` 를 쓰면
segment/local prior 항을 끄고 `H_source_to_target[t,c] = pi_source[c]` 만 적용한다.
Runner default는 v6 main에 맞춰 `--video-prior-no-reliability` 상태다. 이때 segment prior는
그대로 `g(t,c)=s(t,c)*q(t)` 를 쓰지만 video prior만 `pi(c)=max_t s(t,c)` 로 계산해서
video-level에서 reliability `q` 를 한 번 더 곱하지 않는다.

Stage2 rerun은 기본적으로 full 25 labels에서 다시 푼다. 보수적 ablation용으로
`--stage2-active-set stage1` 또는 `--exclude-stage1-sparse` 를 쓰면 target modality의
stage1 active label (`W_stage1(t,c) > eps`) 밖 class에 큰 inactive penalty를 부여한다.
v6 main path는 이 restriction을 쓰지 않고 `--stage2-active-set all` 을 사용한다.

## v5.2 prediction protocol

Stage1과 Stage2는 같은 scoring 함수를 사용한다.

```python
W -> active z-score over nonzero coefficients
K = ||W||_0
T = clip(K / K0, Tmin, Tmax)
p = sigmoid(z / T)
y = 1[p > tau]
```

기본값:
- `tau = 0.75`
- `K0 = 16`
- `Tmin = 0.25`
- `Tmax = 1.25`
- exact-zero weights are rejected candidates and excluded from z-score statistics
- post-hoc min-duration decoding: per `(video, class)`, predictions active for fewer than `--pred-min-duration` segments are removed. Runner default and v6 main use `--pred-min-duration 2`.

구현 위치:
- common scorer: `avvp_stage12.metrics.score_sparse_weights`
- runner/reporting: `run_llp_stage12.py`, `avvp_stage12.reporting`
- table candidates: `scripts/build_main_table_candidates.py`
- sensitivity evaluator: `scripts/evaluate_active_count_temperature.py`

## Runner

v6 main configuration:

```bash
/home/jaemo/AVVP_stage12_clean/scripts/reproduce_main_v6.sh
```

Equivalent explicit command:

```bash
/home/jaemo/miniconda3/envs/av2a_fresh/bin/python \
  /home/jaemo/AVVP_stage12_clean/run_llp_stage12.py \
  --out-dir /home/jaemo/AVVP_stage12_clean/results/main_v6_lam0p3_k4_e1_esc50_noq_mindur2 \
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
  --device cuda
```

```bash
/home/jaemo/miniconda3/envs/av2a_fresh/bin/python \
  /home/jaemo/AVVP_stage12_clean/run_llp_stage12.py \
  --out-dir /home/jaemo/AVVP_stage12_clean/results/default \
  --device cuda
```

LLP mean vs reference mean 비교용 sweep 예시:

```bash
/home/jaemo/miniconda3/envs/av2a_fresh/bin/python \
  /home/jaemo/AVVP_stage12_clean/sweep_lambda.py \
  --out-root /home/jaemo/AVVP_stage12_clean/results/sweep_lambda_extmean \
  --mean-source external \
  --device cuda
```

FSD50K train mean을 새로 만들 때는 아래 두 스크립트를 쓴다.

```bash
/home/jaemo/miniconda3/envs/av2a_fresh/bin/python \
  /home/jaemo/AVVP_stage12_clean/scripts/build_fsd50k_train_manifest.py \
  --ground-truth-csv /mnt/hdd4tb/jaemo/FSD50K/FSD50K.ground_truth/dev.csv \
  --audio-root /mnt/hdd4tb/jaemo/FSD50K/FSD50K.dev_audio \
  --out-csv /mnt/hdd4tb/jaemo/FSD50K/fsd50k_train_manifest.csv
```

```bash
/home/jaemo/miniconda3/envs/av2a_fresh/bin/python \
  /home/jaemo/AVVP_stage12_clean/scripts/extract_audio_mean_from_manifest.py \
  --manifest-csv /mnt/hdd4tb/jaemo/FSD50K/fsd50k_train_manifest.csv \
  --dataset-tag fsd50ktrain \
  --device cuda
```

새 mean을 stage12 sweep에 넣으려면 `--audio-mean-path` 로 직접 넘기면 된다.

```bash
/home/jaemo/miniconda3/envs/av2a_fresh/bin/python \
  /home/jaemo/AVVP_stage12_clean/sweep_lambda.py \
  --out-root /home/jaemo/AVVP_stage12_clean/results/sweep_lambda_fsd50k \
  --mean-source external \
  --audio-mean-path /mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clap_HTSAT-tiny_audio_fsd50ktrain_NXXXXX.npy \
  --device cuda
```

`sweep_lambda.py` 의 F1 plot에는 AV2A pristine baseline을 dotted horizontal line으로 같이 그린다.
기본 metrics는 `/home/jaemo/AV2A_pristine/runs/llp_clipclap_20260420_l2norm_full/per_class_metrics.json` 이며,
다른 `main.py` 결과를 쓰려면 `--av2a-metrics-path /path/to/per_class_metrics.json`,
baseline 없이 보려면 `--no-av2a-baseline` 을 붙인다.
사용한 baseline 값은 sweep output 아래 `av2a_baseline.json` 으로도 저장된다.
또한 cached segment feature와 v25 prompt prototype으로 zero-shot CLAP/CLIP baseline을 재계산해서
F1 plot에 dash-dot horizontal line으로 추가한다.
규칙은 `raw cosine -> per-segment class-axis z-score -> sigmoid -> fixed threshold 0.75` 이며,
값은 `zs_baseline.json` 으로 저장된다. 끄려면 `--no-zs-baseline` 을 붙인다.

주요 저장물:
- `W_a_stage1.npy`, `W_v_stage1.npy`
- `recon_center_a_stage1.npy`, `recon_center_v_stage1.npy`
- `P_a.npy`, `P_v.npy`
- `sparse_confidence_a.npy`, `sparse_confidence_v.npy`
- `reconstruction_quality_a.npy`, `reconstruction_quality_v.npy`
- `reliable_confidence_a.npy`, `reliable_confidence_v.npy`
- `video_prior_a.npy`, `video_prior_v.npy`
- `reliability_a.npy`, `reliability_v.npy`
- `local_support_a.npy`, `local_support_v.npy`
- `plausibility_a.npy`, `plausibility_v.npy`
- `H_v_to_a.npy`, `H_a_to_v.npy`
- `penalty_scale_a.npy`, `penalty_scale_v.npy`
- `lambda_a_weighted.npy`, `lambda_v_weighted.npy`
- `W_a_stage2.npy`, `W_v_stage2.npy`
- `meta.json`

가시화 / 평가 저장물:
- `metrics.json`: stage별 F1 요약
- `metrics_stage1.json`, `metrics_stage2.json`: per-class precision/recall/F1 포함 상세 지표
- `scores_preds_stage1.npz`, `scores_preds_stage2.npz`: v5.2 scores, binary pred, `K`, `T`, GT
- `segment_details_stage1.txt`, `segment_details_stage2.txt`: video/segment/class별 GT, pred, score, raw W

Stage cutoff / detail 옵션:

```bash
/home/jaemo/miniconda3/envs/av2a_fresh/bin/python \
  /home/jaemo/AVVP_stage12_clean/run_llp_stage12.py \
  --out-dir /home/jaemo/AVVP_stage12_clean/results/external_mean \
  --mean-source external \
  --audio-mean-path /path/to/audio_mean.npy \
  --visual-mean-path /path/to/visual_mean.npy \
  --max-stage 2 \
  --score-thr 0.75 \
  --device cuda
```

- `--max-stage 1`: independent stage1까지만 실행하고 `W_*_stage1`, `P_*`, stage1 metrics/detail만 저장
- `--max-stage 2`: v5 cross-modal prior `H` 생성 후 weighted stage2까지 실행
- `--kappa`: local support amplification strength. 기본 `4.0`
- `--eta`: cross-modal prior strength. 기본 `1.0`; `0`이면 stage1로 환원
- `--rho-min`, `--rho-max`: selection cost clipping bound. 기본 `0.1`, `1.0`
- `--stage2-prior-mode full|video`: 기본 `full` 은 `video-level prior * segment prior`; `video` 는 video-level prior만 사용
- `--video-prior-no-reliability`: video prior를 `max_t g(t,c)` 대신 `max_t s(t,c)` 로 계산. 기본값이며 v6 main에서 사용
- `--video-prior-use-reliability`: v5 방식처럼 video prior에 reliability `q`를 포함
- `--stage2-active-set all|stage1`: 기본 `all`. `stage1` 은 target stage1 active label 밖 class를 stage2 rerun에서 사실상 제외
- `--score-mode adaptive_k|fixed_t`: 기본 `adaptive_k` 는 v5.2 `T=clip(K/K0,Tmin,Tmax)`, `fixed_t` 는 같은 active z-score에서 `T=1`
- `--score-k0`, `--score-t-min`, `--score-t-max`: 기본 `16`, `0.25`, `1.25`
- `--pred-min-duration N`: post-hoc decoding filter. per `(video,class)` active segment count가 N 미만이면 prediction 제거. 기본 `2`; 끄려면 `1`
- score normalization 기본값은 `W=0` class를 mean/std 계산에서 제외한다. 이전 방식처럼 zero까지 포함하려면 `--score-include-zero`를 붙인다.
- `--no-details`: 큰 `segment_details_*.txt` 생성을 끔. sweep에서는 이 옵션을 자동으로 사용한다.
- `--detail-max-videos N`: detail txt를 앞 N개 video만 저장. `0`이면 전체 저장
- `--detail-all-classes`: segment마다 25개 class 전체를 저장. 기본은 GT/pred active + top-k만 저장

## Notes

- solver는 batched GPU FISTA 기본값을 쓴다. objective는 draft의 non-negative Lasso와 동일하다.
- weighted stage는 sklearn column-rescale 대신, 같은 objective를 직접 푸는 weighted proximal update로 구현했다.
- v5.2부터 sparse prediction helper는 `avvp_stage12/metrics.py` 의 `score_sparse_weights` 로 고정한다.
