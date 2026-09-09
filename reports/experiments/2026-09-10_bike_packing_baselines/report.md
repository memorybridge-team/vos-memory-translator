# SAM 2 cached baseline suite — bike-packing

> **Pilot:** DAVIS val 한 영상·한 객체·한 switch의 partial evaluation이며 전체
> DAVIS benchmark 결과가 아니다.

## 목적

동일한 prepared-case cache와 SAM 2.1 Large target에서 Direct Copy, Target Reset,
Last-Mask, Replay-k, Full Replay를 실제로 비교했다. 모든 방법은 switch 이후 같은
54개 frame을 처리하며, 공식 DAVIS 평가는 semi-supervised protocol에 맞춰 마지막
video frame을 제외한 frames 15–67에 적용했다.

- Sequence: `bike-packing`
- Object: `1`
- Switch frame: `14`
- Source/target: SAM 2.1 Tiny → Large
- Seed: `7`
- GPU: NVIDIA L4 23GB
- DAVIS evaluator: `davisvideochallenge/davis2017-evaluation@ac7c43fca936f9722837b7fbd337d284ba37004b`

## 입력 정의

- Direct Copy: source의 complete canonical state를 shape 변경 없이 target에 주입
- Target Reset: SAM 2 객체 등록을 위해 switch frame에 빈 mask만 입력하는 proxy
- Last-Mask: source가 switch frame에서 예측한 mask로 target을 초기화
- Replay-k: 최근 k개 frame 구간의 시작점에서 source 예측 mask를 넣고 target이
  switch까지 다시 처리
- Full Replay: DAVIS 첫-frame GT에서 시작해 target이 frame 0–14를 전부 처리

Target Reset, Last-Mask, Replay-k는 DAVIS GT를 모델 입력으로 사용하지 않는다.
Full Replay만 표준 first-frame GT prompt를 사용한다.

## 결과

| 방법 | 과거 처리 frame | J&F | Large-native IoU | 전체 runner 시간 | 과거 backbone 호출 |
|---|---:|---:|---:|---:|---:|
| Direct Copy | 0 | 0.000000 | 0.000000 | 30.511s | 0 |
| Target Reset | 1 | 0.000000 | 0.000000 | 29.672s | 1 |
| Last-Mask | 1 | 0.857995 | 0.973530 | 28.439s | 1 |
| Replay-1 | 1 | 0.857995 | 0.973530 | 28.417s | 1 |
| Replay-2 | 2 | 0.859054 | 0.978674 | 28.847s | 2 |
| Replay-4 | 4 | 0.857450 | 0.977736 | 29.684s | 4 |
| Full Replay / Large-native | 15 | 0.861535 | 1.000000 | 34.490s | 15 |

Last-Mask와 Replay-1 결과가 frame별로 정확히 같아 정의와 구현의 sanity check를
통과했다.

## 해석

1. Direct Copy와 빈-state proxy는 이 case에서 완전히 실패했다. 같은 tensor shape나
   객체 슬롯만으로 target이 이어서 동작하지는 않는다.
2. Last-Mask는 Full Replay보다 J&F가 0.003540 낮을 뿐이다. 이 쉬운 case만으로는
   memory translator의 필요성을 주장할 수 없다.
3. Replay-2는 Last-Mask보다 0.001060 높지만 Replay-4는 오히려 0.000544 낮다.
   더 긴 replay가 항상 성능을 높이지 않는다는 one-case 관찰이다.
4. Full Replay와 Last-Mask의 전체 runner 시간 차이는 약 6.05초다. 다만 모든 방법이
   공통으로 future 54 frames를 처리하고 model/video initialization도 포함하므로,
   handoff 비용 비교에는 wall time과 함께 switch 이전 backbone 호출 15회 대 1회를
   봐야 한다.
5. 다음 핵심 gate는 occlusion/reappearance/fast-motion으로 태그된 여러 case다.
   그곳에서도 Last-Mask가 강하면 CMMT의 필요성이 약해지고, 무너지는 구간에서
   translator 또는 translation+short replay가 회복해야 연구 가치가 생긴다.

## 산출물과 보존 정책

- `summary.json`: machine-readable 수치
- `summary.md`: 자동 생성 표
- RunPod raw suite: `/workspace/CMMT/outputs/baseline_suites/bike-packing_obj1_switch14`
- GitHub Pages gallery: `docs/experiments/2026-09-10-bike-packing-baselines/`

Raw mask와 원해상도 378MB 비교 자료는 Network Volume에 유지한다. Git에는 22MB로
압축한 전체-frame gallery와 작은 수치·보고서만 포함한다.
