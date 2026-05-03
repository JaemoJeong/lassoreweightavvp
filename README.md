# AVVP Stage1/2 Clean Implementation

이 폴더는 `/home/jaemo/AVVP` 안의 기존 실험 코드를 참고해서,
LLP-25 기반 **stage1 independent decomposition** 과
**AVVP_Paper_Draft_KR_v5 Method** 의 cross-modal prior 기반
hint-weighted sparse decomposition을 깔끔하게 다시 구현한 버전이다.

현재 범위:
- stage1: modality별 centered non-negative Lasso decomposition
- stage2: reliability-weighted sparse confidence `g(t,c)` 에서 segment prior `h(t,c)` 와 video prior `pi(c)` 를 대칭적으로 구성
- stage3: `lambda_c(t) = lambda_base * clip(exp(-eta * H(t,c)), rho_min, rho_max)` 로 weighted Lasso 재실행
- stage4: fixed-threshold AVVP scoring + detail report

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
  - audio default: `/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clap_HTSAT-tiny_audio_dcase2017train_N1632.npy`

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
pi_m[c] = max_t g_m[t, c]
H_source_to_target[t, c] = pi_source[c] * (1 + kappa * g_source[t, c])
lambda_target[t, c] = lambda_base * clip(exp(-eta * H_source_to_target[t, c]), rho_min, rho_max)
```

audio reweight에는 visual evidence를, visual reweight에는 audio evidence를 넣는다.
`--stage2-prior-mode video` 또는 `--video-prior-only` 를 쓰면
segment/local prior 항을 끄고 `H_source_to_target[t,c] = pi_source[c]` 만 적용한다.

## Runner

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
- `scores_preds_stage1.npz`, `scores_preds_stage2.npz`: `sigmoid(z-score(W over class axis))` scores, binary pred, GT
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
- `--kappa`: local support amplification strength. 기본 `1.0`
- `--eta`: cross-modal prior strength. 기본 `1.0`; `0`이면 stage1로 환원
- `--rho-min`, `--rho-max`: selection cost clipping bound. 기본 `0.1`, `1.0`
- `--stage2-prior-mode full|video`: 기본 `full` 은 `video-level prior * segment prior`; `video` 는 video-level prior만 사용
- score normalization 기본값은 `W=0` class를 mean/std 계산에서 제외한다. 이전 방식처럼 zero까지 포함하려면 `--score-include-zero`를 붙인다.
- `--no-details`: 큰 `segment_details_*.txt` 생성을 끔. sweep에서는 이 옵션을 자동으로 사용한다.
- `--detail-max-videos N`: detail txt를 앞 N개 video만 저장. `0`이면 전체 저장
- `--detail-all-classes`: segment마다 25개 class 전체를 저장. 기본은 GT/pred active + top-k만 저장

## Notes

- solver는 batched GPU FISTA 기본값을 쓴다. objective는 draft의 non-negative Lasso와 동일하다.
- weighted stage는 sklearn column-rescale 대신, 같은 objective를 직접 푸는 weighted proximal update로 구현했다.
- later experiment에서 threshold / recon-aware scoring을 붙일 수 있도록 helper는 `avvp_stage12/metrics.py` 에 분리해뒀다.
