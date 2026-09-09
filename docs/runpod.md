# RunPod GPU 실행 가이드

이 프로젝트는 RunPod 비용 때문에 연구에 필요한 데이터·baseline·반복 수를
축소하지 않습니다. 대신 GPU와 CPU 작업을 분리하고, 작은 smoke에서 실행 오류를
먼저 제거한 다음 동일한 protocol로 정식 실험을 확장합니다.

## 현재 검증된 환경

- GPU: NVIDIA A40 48GB
- Persistent volume: `/workspace`
- 프로젝트: `/workspace/CMMT`
- 공식 SAM 2 checkout: `/workspace/CMMT/.external/sam2`
- SAM 2.1 Tiny/Large checkpoint: `/workspace/CMMT/checkpoints/`
- DAVIS 2017: `/workspace/CMMT/data/DAVIS`
- 원시 실행 결과: `/workspace/CMMT/outputs/`

Checkpoint, dataset, raw state와 SSH key는 Git에 올리지 않습니다.

## 최초 구성

```bash
cd /workspace
git clone --branch kim/exp-sam2-state-translator \
  https://github.com/memorybridge-team/vos-memory-translator.git CMMT
cd /workspace/CMMT
bash scripts/runpod_bootstrap.sh /workspace/CMMT
```

기존 checkout에서는 새 실험 전에 작업 브랜치를 갱신하고 전체 test를 실행합니다.

```bash
cd /workspace/CMMT
git switch kim/exp-sam2-state-translator
git pull --ff-only
.venv/bin/python -m pytest -q
```

## 자원 사용 원칙

GPU에서 수행:

- 공식 checkpoint inference
- Tiny/Large paired-state 수집
- MLP/attention Translator 학습
- downstream rollout과 latency/VRAM 측정

CPU에서 수행:

- dataset manifest와 split 검증
- Ridge/OLS fit이 메모리에 맞는 경우
- DAVIS metric 집계
- 그래프, MP4와 HTML gallery 생성
- 보고서·Git 산출물 구성

GPU가 0%여도 Pod가 켜져 있으면 요금이 발생할 수 있으므로 실행 상태는 기록합니다.
그러나 비용을 줄이려고 필요한 실험 case를 제거하지는 않습니다.

학습 중 외부 object storage의 JPEG/PNG를 직접 읽지 않습니다. 현재 case의 dataset과
checkpoint를 `/workspace/CMMT`에 먼저 준비하고 GPU 작업이 끝난 뒤 결과를
외부 저장소로 올리는 hot/cold 구조를 사용합니다. 자세한 기준은
[`storage_pipeline.md`](storage_pipeline.md)를 따릅니다.

## 실행 순서

1. 1개 case smoke로 경로·checkpoint·state contract·artifact 생성을 확인합니다.
2. 고정 manifest 전체에서 Target Reset, Last-Mask, Replay-k, Full Replay,
   Direct Transfer를 같은 evaluator로 실행합니다.
3. train video에서 paired state를 수집하고 validation/test video와 분리합니다.
4. Ridge와 component-wise MLP를 학습합니다.
5. post-switch J&F, identity, recovery, latency·VRAM·bytes를 집계합니다.
6. 전체 영상 gallery와 실패 case를 검수한 뒤 보고서를 Git에 반영합니다.

현재 유효한 전체 계획과 Go/No-Go 기준은
[`experimental_plan.md`](experimental_plan.md)를 따릅니다.

## Git에 가져올 결과

- 실행 config, seed, upstream/checkpoint 식별자
- case별 raw metric JSON과 요약 Markdown
- 대표 실패·성공 PNG
- 공개 검토용으로 압축한 MP4/HTML gallery
- 재현 명령과 알려진 한계

대량 binary mask, raw state tensor, dataset과 checkpoint는 RunPod persistent
volume에 보관하고 Git에는 위치·hash·schema만 기록합니다.

## 현재 공개 Pilot

2026-09-08 Tiny→Large DAVIS pilot의 전체 83-frame 결과는
[GitHub Pages gallery](https://memorybridge-team.github.io/vos-memory-translator/experiments/2026-09-08-davis-handoff/)에서
확인할 수 있습니다. 이는 한 영상·한 객체·한 switch의 제한적 pilot이며 전체
DAVIS benchmark 결과가 아닙니다.
