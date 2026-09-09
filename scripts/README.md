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
