# SAM 2 cached baseline hard case — india reappearance

> **Pilot:** DAVIS 2017 val의 한 영상·한 객체·한 switch에 대한 partial
> evaluation이다. 전체 DAVIS benchmark 결과가 아니다.

## 목적

`india` object 3가 완전히 사라진 frame 35에서 SAM 2.1 Tiny를 Large로 바꾼 뒤,
과거 상태 없이도 다시 등장한 같은 객체를 이어서 찾을 수 있는지 확인했다. 비교
방법은 Direct Copy, Target Reset, Last-Mask, Replay-1/2/4, Full Replay다.

- Sequence: `india`
- Object: `3`
- Switch frame: `35`
- 공식 평가 frame: `36–79` (마지막 video frame 제외)
- Source/target: SAM 2.1 Tiny → Large
- Seed: `7`
- GPU: NVIDIA L4 23GB
- SAM 2 commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- DAVIS evaluator: `davisvideochallenge/davis2017-evaluation@ac7c43fca936f9722837b7fbd337d284ba37004b`

## 왜 hard case인가

Object 3의 DAVIS GT는 frame 35에서 0픽셀이고 frame 36에서 11,393픽셀로 다시
나타난다. frame 46–54에도 다시 완전히 사라진 뒤 frame 55부터 재등장한다.

Tiny source가 만든 prompt mask의 양성 픽셀 수도 다음과 같았다.

| Frame | 용도 | 양성 픽셀 |
|---:|---|---:|
| 32 | Replay-4 시작 | 25,515 |
| 34 | Replay-2 시작 | 2,298 |
| 35 | Last-Mask / Replay-1 | 0 |

따라서 Last-Mask는 물체의 모습이나 위치 단서를 target에 전달하지 못한다. 반면
Replay-2/4는 물체가 아직 보이던 과거 frame에서 Large를 시작할 수 있다.

## 결과

| 방법 | 과거 처리 frame | 전체 J&F | GT-visible J&F | Large-native IoU | 전체 runner 시간 | 과거 backbone 호출 |
|---|---:|---:|---:|---:|---:|---:|
| Direct Copy | 0 | 0.204545 | 0.000000 | 0.222222 | 27.875s | 0 |
| Target Reset | 1 | 0.204545 | 0.000000 | 0.222222 | 26.530s | 1 |
| Last-Mask | 1 | 0.204545 | 0.000000 | 0.222222 | 25.954s | 1 |
| Replay-1 | 1 | 0.204545 | 0.000000 | 0.222222 | 25.526s | 1 |
| Replay-2 | 2 | 0.552702 | 0.437682 | 0.584968 | 26.243s | 2 |
| Replay-4 | 4 | 0.868743 | 0.834992 | 0.956255 | 27.039s | 4 |
| Full Replay / Large-native | 36 | 0.895240 | 0.868302 | 1.000000 | 39.630s | 36 |

모든 방법은 같은 45개 future frame을 처리했다. Last-Mask와 Replay-1은 frame별로
정확히 같아 구현 sanity check를 통과했다.

## 중요한 평가 해석

전체 J&F만 보면 실패한 방법도 0.204545점을 얻는다. 그러나 이는 GT가 비어 있는
9개 frame에서 빈 예측이 J=1, F=1로 채점됐기 때문이다. GT에 object가 실제 존재하는
35개 frame만 평균하면 Direct/Reset/Last-Mask/Replay-1은 모두 0이다. 이 사례부터
전체 J&F와 `GT-visible J&F`를 항상 함께 보고한다.

## 판단

1. Last-Mask는 물체가 switch 시점에 사라지면 재등장을 전혀 회복하지 못했다.
2. Replay-2는 일부 회복하지만 visible J&F 0.438로 불충분하다.
3. Replay-4는 4개 과거 frame만 다시 처리하고 Full Replay의 visible J&F보다
   0.03331 낮은 0.835를 얻었다. 이 사례에서는 translated memory + short replay가
   강한 설계 후보임을 보여준다.
4. Direct Copy는 source state 36개를 전달하고도 visible J&F 0이다. shape가 같은
   Tiny/Large state도 의미 공간이 정렬됐다고 볼 수 없으며 learned translator가
   필요한 이유다.
5. 아직 one-case pilot이므로 translator 우월성의 증거는 아니다. 다음 단계는 고정
   rare-event subset 전체에서 baseline 분포를 만든 뒤, 같은 split으로 learned
   translator와 translator+short-replay를 학습·평가하는 것이다.

## 산출물

- `summary.json`: machine-readable 수치
- `summary.md`: 자동 생성 표
- RunPod raw suite: `/workspace/CMMT/outputs/baseline_suites/india_obj3_switch35`
- RunPod case cache: `/workspace/CMMT/outputs/case_cache/india_obj3_switch35.pt`
- GitHub Pages gallery: `docs/experiments/2026-09-10-india-reappearance-baselines/`

Raw mask와 273MB prepared cache는 Network Volume에 유지한다. Git에는 11.8MB의
44-frame×7-method 선택형 gallery와 작은 수치·보고서만 포함한다.
