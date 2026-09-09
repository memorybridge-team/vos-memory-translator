# Scripts

paired-state 생성, translator 학습, baseline 평가와 결과 집계를 위한 얇은 실행 진입점을 둡니다. 핵심 로직은 재사용과 테스트가 가능하도록 `src/vos_memory_inspector/`에 구현합니다.

- `runpod_bootstrap.sh`: 고정된 SAM 2 checkout, CUDA extension, Tiny/Large checkpoint를 준비합니다.
- `runpod_direct_smoke.sh`: Tiny→Direct Copy→Large smoke를 실행하고 Git/VS Code에서 볼 수 있는 PNG·JSON·Markdown 결과를 만듭니다.
- `make_synthetic_video.py`: 외부 데이터셋 없이 실행 경로를 확인할 작은 영상을 만듭니다.
- `run_tiny_large_probe.py`: Tiny/Large의 canonical state contract를 순차 비교합니다.

고정 DAVIS case manifest는 설치 후 다음 명령으로 생성합니다. GT는 switch event
선택과 평가 tag에만 사용하며 모델 입력에는 사용하지 않습니다.

```bash
cmmt-davis-build-manifest \
  --root /workspace/CMMT/data/DAVIS \
  --split val \
  --output /workspace/CMMT/outputs/manifests/davis2017_val_phase1.json
```

여러 baseline이 같은 source/oracle을 다시 계산하지 않도록 case reference를 한 번만
준비할 수 있습니다. `.pt`와 `.pt.sha256`은 `outputs/`에 두며 Git에 올리지 않습니다.

```bash
cmmt-sam2-prepare-case \
  --sam2-repo /workspace/CMMT/.external/sam2 \
  --source-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --source-checkpoint /workspace/CMMT/checkpoints/sam2.1_hiera_tiny.pt \
  --source-model-id sam2.1-hiera-tiny \
  --target-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --target-checkpoint /workspace/CMMT/checkpoints/sam2.1_hiera_large.pt \
  --target-model-id sam2.1-hiera-large \
  --video-dir /workspace/CMMT/data/DAVIS/JPEGImages/480p/bmx-bumps \
  --prompt-mask /workspace/CMMT/data/DAVIS/Annotations/480p/bmx-bumps/00000.png \
  --object-id 1 --switch-frame 6 \
  --output /workspace/CMMT/outputs/case_cache/bmx-bumps_obj1_switch6.pt
```

준비된 cache에서는 source와 oracle을 다시 실행하지 않고 candidate continuation만
실행합니다.

```bash
cmmt-sam2-cached-handoff \
  --case-cache /workspace/CMMT/outputs/case_cache/bmx-bumps_obj1_switch6.pt \
  --sam2-repo /workspace/CMMT/.external/sam2 \
  --target-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --target-checkpoint /workspace/CMMT/checkpoints/sam2.1_hiera_large.pt \
  --target-model-id sam2.1-hiera-large \
  --video-dir /workspace/CMMT/data/DAVIS/JPEGImages/480p/bmx-bumps \
  --translator direct
```
