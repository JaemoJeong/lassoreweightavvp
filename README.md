# AVVP Stage1/2 Clean Implementation

이 폴더는 `/home/jaemo/AVVP` 안의 기존 실험 코드를 참고해서,
LLP-25 기반 **stage1 independent decomposition** 과
**stage2 video-level presence + video-level embedding absence** 만을 사용한
cross-modal reweight 파이프라인을 깔끔하게 다시 구현한 버전이다.

현재 범위:
- stage1: modality별 centered non-negative Lasso decomposition
- stage2: `P = max_t w(t)` 와 video-level sparse decomposition 기반 `A`
- stage3: `beta * P - gamma * A` evidence로 weighted Lasso 재실행
- stage4: full AVVP scoring 실험 전용 helper만 준비, 기본 runner에는 강제하지 않음

이번 구현에서 의도적으로 제외한 것:
- segment-level confidence hint
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
video-level centering mean 은 segment mean 과 따로 계산한다.

실험용으로 `run_llp_stage12.py` 는 mean source를 둘 다 지원한다.
- `--mean-source llp`: LLP test feature mean
- `--mean-source external`: precomputed backbone reference mean
  - visual default: `/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clip_ViT-L-14_image_mscoco_train_N118287.npy`
  - audio default: `/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clap_HTSAT-tiny_audio_dcase2017train_N1632.npy`

## Weighted hint

이번 버전은 confidence 없이 아래 두 항만 쓴다.

```python
P[c] = max_t w_stage1[t, c]
A[c] = max(0, max_k w_video[k] - w_video[c])
e = beta * P - gamma * A
lambda_c = lambda_base * max(lambda_min_factor, 1 - e)
```

audio reweight에는 visual evidence를, visual reweight에는 audio evidence를 넣는다.

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

주요 저장물:
- `W_a_stage1.npy`, `W_v_stage1.npy`
- `P_a.npy`, `P_v.npy`
- `W_a_video.npy`, `W_v_video.npy`
- `A_a.npy`, `A_v.npy`
- `lambda_a_weighted.npy`, `lambda_v_weighted.npy`
- `W_a_stage2.npy`, `W_v_stage2.npy`
- `meta.json`

## Notes

- solver는 batched GPU FISTA 기본값을 쓴다. objective는 draft의 non-negative Lasso와 동일하다.
- weighted stage는 sklearn column-rescale 대신, 같은 objective를 직접 푸는 weighted proximal update로 구현했다.
- later experiment에서 threshold / recon-aware scoring을 붙일 수 있도록 helper는 `avvp_stage12/metrics.py` 에 분리해뒀다.
