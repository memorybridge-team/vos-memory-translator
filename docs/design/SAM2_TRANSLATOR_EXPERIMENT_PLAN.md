# SAM 2 Translator 단계별 실행 계획

작성일: 2026-09-07
첫 controlled pair: SAM 2.1 Tiny → Large
기준 upstream: `facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4`

## 목표

SAM 2.1 Tiny가 frame `1...t`까지 축적한 state `b_t`를 translator로
Large-compatible state `â_t`로 바꾸고, Large가 과거 frame을 다시 처리하지
않은 채 frame `t+1`부터 추론을 이어 가는 end-to-end handoff를 검증한다.

## 단계와 통과 기준

### 0. 실행 자산 준비

- 공식 SAM 2 저장소를 별도 local checkout으로 clone하고 위 commit으로 고정한다.
- SAM 2.1 Tiny/Large checkpoint를 공식 링크에서 내려받아 Git 비추적 경로에 둔다.
- DAVIS 2017 val을 내려받고 sequence/frame/annotation 정합성을 검사한다.
- Python/PyTorch/CUDA 환경과 seed를 기록한다.

통과 기준:

- `verify_sam2_checkout`가 고정 commit과 private state contract를 통과한다.
- Tiny/Large checkpoint 파일이 존재하고 모델 construction이 성공한다.
- 선택한 DAVIS val sequence의 RGB frame과 첫-frame GT mask가 일치한다.

### 1. 동일-checkpoint export→inject round-trip

같은 B checkpoint가 prefix `1...t`를 처리해 만든 state를 export한 뒤, 별도의
B predictor에 materialize/inject한다. 주입된 B는 prefix를 재처리하지 않고
`t+1`부터 실행한다.

통과 기준:

- object registry, cond/non-cond history, prompt history, frame metadata가 보존된다.
- positional encoding은 source tensor 복사가 아니라 target 규칙으로 생성된다.
- injection 뒤 과거-frame encoder 호출 수가 0이다.
- 원본 B continuation과 round-trip B continuation의 mask 차이가 허용 범위 안이다.

### 2. Tiny/Large paired-state 수집

동일한 video, object prompt, switch frame을 Tiny와 Large에 각각 입력한다.

```text
Tiny(prefix 1...t)  -> b_t
Large(prefix 1...t) -> a_t  (target-native teacher/oracle state)
```

통과 기준:

- video/object/frame/conditioning role이 정확히 대응한다.
- train/validation/test video가 분리되어 있다.
- 미래 frame과 test state가 translator 학습에 들어가지 않는다.

### 3. Translator baseline 학습

다음 순서로 복잡도를 높인다.

1. Direct Copy
2. component-wise Ridge/OLS
3. component-wise Linear
4. residual two-layer MLP

`maskmem_features`, `obj_ptr`, `object_score_logits`는 별도 head로 다룬다.
`maskmem_pos_enc`는 target에서 재생성하고 object ID, frame index, slot order,
conditioning flag, validity는 그대로 보존한다.

통과 기준:

- held-out state에서 MSE/cosine/relative error를 기록한다.
- Direct 대비 개선과 parameter 수, translator latency를 함께 기록한다.
- tensor metric만으로 downstream 성공을 선언하지 않는다.

### 4. 실제 A→Translator→B continuation

```text
Tiny(prefix 1...t) -> b_t -> T -> â_t
                                  |
                                  v
                         Large에 injection
                                  |
                                  v
                         frame t+1... 추론
```

Large가 출력한 post-switch mask를 dataset GT mask와 비교한다. Large가 prefix를
직접 처리해 얻은 continuation은 target-native oracle로만 사용하며 translator
입력에는 사용하지 않는다.

통과 기준:

- frame `t+1`부터 실제 mask 출력이 생성된다.
- J&F, switch 후 1/5/20-frame 성능, identity break, recovery length를 기록한다.
- Target Reset, Last-Mask, replay-1/2/4, Direct, Full Replay/Oracle와 비교한다.

### 5. 시스템 비용과 Go/No-Go

- handoff latency, replay latency, translator latency, 전송 byte, peak VRAM을 측정한다.
- hard occlusion/reappearance slice에서 Last-Mask와 replay-k를 우선 비교한다.
- Tiny→Large가 정보 병목으로 실패하면 translation+replay-1/2/4를 평가한다.

Go 판단은 learned translator 또는 hybrid가 hard condition에서 downstream 성능을
개선하면서 replay frontier에 Pareto-dominated되지 않을 때 내린다. Direct Copy가
oracle에 가까우면 same-family learned translator를 확대하지 않고 그 결과를
baseline으로 남긴 뒤 cross-architecture pair로 진행한다.

## 정답과 평가 대상의 구분

- `a_t`: Large가 prefix를 직접 처리해 만든 내부 teacher/oracle state. Translator의
  state-level 학습·진단에 사용한다.
- dataset GT mask: post-switch segmentation의 실제 정답. 최종 J&F 평가에 사용한다.
- target-native continuation: Full Replay/Oracle 성능 상한선. 실제 배포 경로에는
  포함되지 않는다.

## 현재 경계

저장소에는 state inspector, canonicalizer, Direct/Ridge/Linear/MLP, synthetic
tests와 Tiny/Large sequential probe가 있다. 실제 checkpoint 기반 target history
injection은 구현되었다. 2026-09-07 CPU synthetic-video smoke에서 Tiny와 Large의
동일-checkpoint round-trip이 각각 미래 mask MSE 0, binary IoU 1.0으로 통과했고,
injection 전·도중 과거-frame backbone 호출은 0회였다.

같은 smoke에서 Tiny→Direct Copy→Large도 end-to-end로 실행됐다. Tensor shape는
일치했지만 spatial-memory cosine은 약 -0.0073, object-pointer cosine은 약
-0.0528이었고, 다음 frame mask의 Large-native 대비 binary IoU는 0이었다. 이는
Direct Copy가 의미적으로 실패할 수 있음을 보여 주는 **단일 synthetic-video
smoke**이며 dataset 성능 결론이 아니다. DAVIS/LVOS/MOSE 기반 translator 학습과
J&F 평가는 아직 수행되지 않았다.
