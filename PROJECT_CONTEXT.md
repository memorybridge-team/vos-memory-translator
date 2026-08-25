# Cross-Model Memory Translator — Project Context

> 새 채팅을 위한 프로젝트 영구 컨텍스트  
> 로컬 스냅샷 기준일: **2026-08-26 (KST)**
> 원자료: 프로젝트 노션 데이터베이스 및 현재까지의 대화  
> 상태 표기: **[확인]** 문헌·코드·기록으로 확인 / **[가설]** 실험 필요 / **[Pilot]** 제한적 예비 결과

## 1. 한 문장 정의

장시간 비디오를 처리하던 stateful model을 중간에 다른 모델로 교체할 때, source model의 누적 temporal memory를 target-compatible memory로 번역하여 **과거 프레임 전체를 replay하지 않고도 target model이 다음 프레임부터 이어서 추론**하게 만드는 연구다.

## 2. 쉬운 설명

비디오 모델은 현재 프레임만 보는 것이 아니라, 지금까지 본 물체의 모습·위치·가림·정체성에 관한 내부 메모리를 쌓는다. 처리 도중 작은 모델에서 큰 모델로, 또는 한 구조에서 다른 구조로 바꾸면 새 모델은 그동안의 과거를 모른다. 가장 확실한 해결책은 과거 프레임을 새 모델로 다시 읽는 것이지만, 긴 영상에서는 느리고 비싸다.

CMMT는 이전 모델의 메모리를 일종의 통역기를 거쳐 새 모델이 이해할 수 있는 메모리로 바꾼다.

```text
과거 프레임 1 ... t
        │
        ▼
source model ── source memory b_t
                         │
                         ▼
                 translator T
                         │
                         ▼
                 target memory â_t
                         │
                         ▼
target model이 frame t+1부터 계속 처리
```

핵심 식은 다음과 같다.

```text
â_t = T(b_t) ≈ a_t
```

- `b_t`: source model이 frame 1...t를 처리하며 만든 state
- `a_t`: target model이 동일한 과거를 직접 처리했을 때의 native/oracle state
- `â_t`: translator가 만든 target-compatible state

성공은 `â_t`와 `a_t`의 원소별 오차가 작은지만으로 판단하지 않는다. `â_t`를 주입한 target model이 이후 프레임에서 일을 잘하고, 전환 비용을 실제로 줄이는지가 기준이다.

## 3. target model은 전달된 기억으로 무엇을 하는가

초기 주 태스크는 Video Object Segmentation/Tracking이다. target model은 전달된 memory를 이용해 다음을 수행한다.

- 다음 프레임에서 같은 객체의 mask를 예측한다.
- 객체가 일부 또는 완전히 가려진 뒤 다시 나타났을 때 동일한 객체임을 복구한다.
- 비슷하게 생긴 distractor 사이에서 identity를 유지한다.
- 모습·크기·자세·조명 변화가 있어도 과거의 object representation과 현재 관측을 연결한다.
- 전환 직후 불안정한 mask가 여러 프레임에 걸쳐 누적되는 것을 줄인다.

단순하고 짧으며 가림이 없는 장면에서는 직전 mask만 넘기는 Last-Mask baseline이 충분히 강할 수 있다. 따라서 과거 memory의 필요성은 occlusion/reappearance, distractor, fast motion, long-term appearance change가 있는 hard scenario에서 입증해야 한다.

## 4. SAM 2에서 고려할 memory/state

**[확인]** SAM 2는 streaming video inference를 위해 memory bank와 object pointer를 사용하는 명시적 temporal-memory 구조를 갖는다. 구현 시 최소한 다음 계열을 구분한다.

- `maskmem_features`: 과거 mask와 영상 특징에서 형성된 spatial memory feature
- `maskmem_pos_enc`: memory token의 공간·시간 위치를 해석하기 위한 positional encoding
- `obj_ptr`: 객체 정체성과 고수준 의미를 압축한 object pointer
- object presence / occlusion 관련 신호: 객체가 현재 존재하거나 보이지 않는 상태에 관한 신호
- conditioning/non-conditioning frame 및 temporal-position metadata
- memory bank의 slot 선택·순서·유효성 정보

주의할 점:

- 이 state를 사람이 읽을 수 있는 사건 목록으로 해석하면 안 된다. 대부분은 학습된 분산 표현이다.
- feature만 번역하고 object pointer, presence, positional metadata를 누락하면 실제 handoff가 불완전할 수 있다.
- source와 target의 layer width, token 수, spatial resolution, slot 정책, object-centric 표현이 다를 수 있으므로 단일 선형 projection만으로 충분하다고 가정하지 않는다.

### 4.1 SAM 2.1 same-family 호환성 — 2026-08-20 확인

**[확인]** 공식 SAM 2.1의 Tiny, Small, Base+, Large는 backbone 규모와 처리 속도는 다르지만, 공개 설정상 video-memory 경계는 매우 강하게 정렬되어 있다.

- backbone 초기 차원은 Tiny/Small 96, Base+ 112, Large 144로 다르다.
- 네 모델 모두 image-encoder neck 및 memory-attention의 `d_model`은 256이다.
- memory-encoder의 출력 차원은 모두 64이고 `num_maskmem`은 모두 7이다.
- object pointer, object-presence score, temporal positional encoding을 사용하는 기본 구조와 관련 옵션도 같다.
- 공식 SAM 2.1 표에서 Tiny→Large는 38.9M→224.4M parameters, 91.2→39.5 FPS로 계산량 차이가 크다(A100 기준).

따라서 SAM 2.1 Tiny↔Large는 **tensor contract가 맞고 representation semantics만 다를 가능성을 시험하는 controlled same-family pair**로 매우 적합하다. 이는 LLM same-family KV transfer의 paired-state 수집, direct-copy, ridge/linear map, 작은 nonlinear translator라는 실험 절차를 적용하기 좋은 조건이다.

단, shape가 같다고 의미 공간까지 같다는 뜻은 아니다. 학습 전 `raw/direct state copy`를 반드시 baseline으로 측정한다. direct copy가 target-native oracle에 거의 도달하면 same-family translator의 필요성은 약해지며, 이 경우 해당 결과를 baseline으로 남기고 SAM 2↔XMem/Cutie 같은 cross-architecture pair로 중심을 옮긴다.

## 5. 연구 질문

1. source memory만으로 target-native state에 가까운 유용한 target state를 만들 수 있는가?
2. 어느 방향이 더 어려운가: Small→Large, Large→Small, cross-architecture?
3. tensor reconstruction이 아니라 downstream error를 줄이는 loss와 translator 구조는 무엇인가?
4. full replay 대비 어느 길이의 영상·어떤 switch 조건에서 latency/compute 이득이 생기는가?
5. Last-Mask, replay-k, dynamic submodel보다 memory translation이 필요한 hard scenario는 무엇인가?
6. 반복 전환에서 state error가 누적되는가? 그렇다면 refresh 또는 short replay가 필요한가?
7. 모델 쌍의 transferability를 학습 전에 예측할 수 있는가?

## 6. 성공 기준

“큰 모델이 만든 `a_t`를 정답으로 두고 작은 모델의 `b_t`를 `a_t` 형태로 변환할 수 있다”는 것은 중요한 학습 설정이지만, 그것만으로 프로젝트 성공은 아니다.

다음을 함께 만족해야 한다.

1. **기능적 연속성**: target이 frame `t+1`부터 reset 없이 정상 동작한다.
2. **후속 성능**: target-native oracle에 가까운 J&F와 identity continuity를 보인다.
3. **강한 baseline 우위**: hard condition에서 Last-Mask와 실용적인 replay-k보다 낫거나, 같은 성능에서 훨씬 싸다.
4. **시스템 이득**: full replay보다 handoff latency, FLOPs, energy 또는 비용이 줄어든다.
5. **양방향성**: Large→Small뿐 아니라 Small→Large의 한계와 가능한 보완책을 정직하게 제시한다.
6. **반복 안정성**: 여러 번의 switch에서도 오차가 통제된다.
7. **일반성**: 가능하면 SAM 2 small↔large를 넘어 SAM 2↔Cutie/XMem 등 cross-architecture pair에서 성립한다.

## 7. 권장 방법론

### 7.1 데이터 생성

동일한 비디오 prefix `1...t`를 frozen source와 frozen target model에 각각 통과시켜 `(b_t, a_t)` state pair를 수집한다. backbone은 우선 고정하고 translator만 학습한다.

### 7.2 translator 후보

난이도 순으로 비교한다.

- direct/no-op transfer 또는 shape adapter
- per-layer/per-slot linear mapping 및 ridge regression
- 얕은 MLP
- token resampler + projection
- attention-based set translator
- component-specific translator: spatial memory, object pointer, presence/metadata를 따로 번역한 뒤 결합
- 모델 쌍이나 상태 유형별 Mixture-of-Translators

translator는 target model의 입력 state 계약을 지켜야 하며, 단순히 같은 shape을 출력하는 것과 target이 사용할 수 있는 state를 출력하는 것을 구분한다.

### 7.3 loss 후보

- feature/state reconstruction loss
- attention-output 또는 readout alignment loss
- target decoder가 내는 후속 mask의 distillation loss
- identity/object-pointer alignment loss
- multi-step rollout loss: switch 이후 여러 프레임의 누적 성능을 직접 최적화
- uncertainty/presence/occlusion consistency loss

**핵심 원칙:** raw reconstruction error가 작아도 downstream-important subspace가 틀리면 실패한다. 반대로 원소별 오차가 다소 커도 후속 prediction이 유지되면 좋은 translator일 수 있다.

### 7.4 정보 병목 대응

Small→Large에서 source가 보존하지 않은 정보를 translator가 무에서 복원할 수는 없다. 다음 선택지를 실험한다.

- translation only
- translation + 최근 `k`개 frame short replay
- uncertainty가 높을 때만 selective replay
- 중요한 event frame 또는 conditioning frame만 replay
- 주기적 target-native refresh

hybrid 방식은 아이디어의 실패가 아니라, full replay를 크게 줄이는 시스템 설계로 정의할 수 있다.

## 8. 필수 baseline

- **Target Reset**: target을 빈 state로 시작
- **Last-Mask**: 직전 mask 또는 최소 prompt만 target에 전달
- **Replay-k**: 최근 `k`개 프레임만 target으로 재처리
- **Full Replay / Target-Native Oracle**: 과거 전체를 target이 처리한 상한선
- **Direct Transfer**: 학습된 translation 없이 가능한 형태 변환만 수행
- **Learned Translator**: 제안 방식
- **Translation + Short Replay**: hybrid
- **RDS 또는 matched-compute dynamic model**: 직접적인 handoff baseline이 아니라 모델 교체를 피하는 scenario-level 대안. 동일한 사용 조건을 구성할 수 있을 때만 별도 비교

가능한 경우 모든 방법을 같은 accuracy 또는 같은 compute budget에서 비교한다.

## 9. 평가 설계

### 9.1 데이터셋

- 표준 성능 확인: DAVIS, YouTube-VOS
- hard/long-term 조건: MOSE, LVOS
- 필요 시 대규모 추가 검증: SA-V

easy benchmark의 평균값만으로 결론을 내리지 않는다. occlusion, reappearance, distractor, camera cut, fast motion, long absence를 별도 slice로 분석한다.

### 9.2 switch scenario

- Small→Large
- Large→Small
- 반복 switch
- 무작위 시점 switch
- event-based switch: occlusion 직전/중간/직후, reappearance, 빠른 움직임, distractor 등장
- 서로 다른 장치 또는 latency budget 사이의 실제 handoff 시나리오

### 9.3 핵심 지표

정확도·연속성:

- J&F 및 J, F
- switch 후 1/5/20 frame 성능
- handoff gap 또는 switching shock
- target-native 성능으로 돌아오는 recovery length
- identity break / ID switch
- temporal consistency

효율:

- handoff latency
- full replay 및 replay-k 대비 처리 시간
- FLOPs/MACs
- peak VRAM과 전달 바이트 수
- translator parameter size
- energy 또는 cloud cost(가능한 경우)

통계:

- 영상별/객체별 결과를 유지
- 평균뿐 아니라 분산과 confidence interval 또는 bootstrap 사용
- switch point를 독립 표본처럼 과도하게 계산하지 않도록 clustered analysis 고려

## 10. RDS(CVPR 2026)와의 관계

**[확인]** RDS는 previous mask와 current feature를 이용한 router가 단일 SAM 2 계열 모델 내부에서 필요한 block을 동적으로 활성화하는 방식이다. 즉, 별도의 small/large model을 교체하고 state를 번역하는 대신 하나의 모델을 dynamic submodel처럼 사용한다.

따라서 RDS는 memory translator의 구조나 학습법을 설계하기 위한 직접 선행연구가 아니다. 다만 “왜 굳이 모델을 바꾸는가? dynamic model을 쓰면 되지 않는가?”라는 문제 설정 공격에 답하기 위한 간접 관련 연구다. 전체 구현을 따라갈 필요는 없으며 problem formulation, routing 범위, 한계와 실험 조건을 중심으로 읽는다.

CMMT의 방어 논리는 다음 조건에서만 강하다.

- edge↔cloud, mobile↔server처럼 서로 다른 장치 사이에 state를 넘겨야 한다.
- 라이선스·전문화·가용성·에너지·메모리 제약 때문에 실제로 다른 모델을 사용해야 한다.
- SAM 2와 Cutie/XMem처럼 architecture 또는 state semantics가 다르다.
- 한 모델의 dynamic blocks만으로는 제공할 수 없는 capability 전환이 필요하다.
- 모델을 교체한 뒤 full history replay가 병목이 된다.

반대로 단일 모델 내부 계산량 조절만 필요한 상황에서는 RDS가 더 단순하고 강한 답일 수 있다. CMMT 논문은 generic adaptive inference를 주장하기보다 **heterogeneous stateful-model handoff**를 명확한 문제로 고정해야 한다.

## 11. 관련 연구에서 가져온 핵심 교훈

### Cross-Model KV Cache Transfer

**[확인]** 서로 다른 LLM 사이의 KV cache를 변환해 re-prefill을 피하는 선행 사례다. 같은 계열 모델에서는 ridge/linear mapping이 유효할 수 있고, 어려운 pair에서는 per-layer/head nonlinear mapping이 크게 개선될 수 있다. 반복 switch에서 drift가 생길 수 있으며, reconstruction보다 downstream-important error placement가 중요하다는 교훈을 준다.

CMMT에 주는 의미:

- cross-model state translation 자체의 큰 아이디어만으로 novelty를 주장하면 안 된다.
- VOS memory는 LLM KV와 달리 spatial/object-centric이며, 이 차이를 방법과 평가에서 실질적으로 활용해야 한다.
- LLM은 긴 prompt re-prefill 비용이 명확하지만 VOS는 Last-Mask가 충분한 경우가 있어, “왜 전체 memory가 필요한가”를 hard scenario로 증명해야 한다.

### Cache-to-Cache

source LLM의 KV semantics를 projection한 뒤 target LLM의 자체 cache와 gated fusion하여 모델 간 직접 semantic communication을 수행한다. 동일 prefix를 처리한 target-native cache를 source cache만으로 재구축하고 다음 step부터 이어가는 문제는 아니다. 따라서 CMMT의 정확한 analogue라기보다 component별 선택적 주입과 `translated source memory + target-native context` hybrid 설계의 참고점이다.

### Model Stitching / Zero-shot Stitching / vec2vec

서로 다른 representation space를 연결하는 adapter, anchor, alignment 방법의 참고군이다. 그러나 CMMT는 정적 feature alignment가 아니라 시간에 따라 누적되고 다음 prediction에 재사용되는 state를 다룬다는 점을 강조한다.

### SAM 2, XMem, Cutie, SAM2Long

- SAM 2: 명시적 streaming memory와 object pointer가 있어 첫 구현에 적합
- XMem: sensory/working/long-term memory 구조의 cross-architecture target 후보
- Cutie: object-centric memory 및 강한 VOS baseline 후보
- SAM2Long: long-video memory selection과 누적 오류 분석 참고

현재 로컬/노션 기록에서 XMem과 SAM2Long의 상세 분석은 아직 비어 있으므로, 해당 내용을 확인된 사실처럼 확장하지 않는다.

## 12. 현재 Pilot 상태와 해석

**[Pilot]** 기존 기록에는 translator, replay-N, Last-Mask를 비교한 예비 실험이 있다.

- 규모: 10개 영상 × 10개 switch point = 100개 사례
- 환경: Apple Metal 기반으로 기록됨
- 조건: 비교적 easy video 중심
- 코드: GPT 생성 코드 기반의 초기 검증
- replay는 정확도를 높이지만 시간이 증가하는 전형적 trade-off를 보였다.
- Last-Mask가 예상보다 강했다.
- Small→Large는 약했고 Large→Small은 상대적으로 유망했다.
- object pointer transfer가 구현되지 않아 완전한 memory handoff 실험으로 보기 어렵다.
- translation + short replay hybrid가 유망할 가능성이 제기됐다.

이 결과는 아이디어를 확정하거나 폐기할 수준의 증거가 아니다. 특히 easy data, 미완성 state transfer, 제한된 pair와 switch 설정 때문에 논문 결론에 직접 사용하면 안 된다.

## 13. 가장 큰 위험과 반증 조건

1. **Last-Mask sufficiency**: hard data에서도 Last-Mask가 translator와 거의 같으면 temporal-memory translation의 필요성이 약하다.
2. **Replay crossover**: 필요한 replay 길이가 짧고 충분히 빠르면 translator의 시스템 가치가 작다.
3. **Information loss**: Small→Large에서 target-native quality를 만들 정보가 source state에 없다.
4. **Architecture specificity**: translator가 특정 SAM 2 checkpoint pair에서만 작동하면 일반성이 약하다.
5. **Error accumulation**: 반복 switch에서 작은 state mismatch가 빠르게 누적된다.
6. **State contract fragility**: 내부 implementation detail에 과도하게 의존해 버전이나 모델 변경에 취약하다.
7. **RDS domination**: 실제 사용 시나리오가 heterogeneous handoff를 요구하지 않으면 dynamic submodel이 더 적절하다.
8. **Evaluation leakage**: target-native state를 교사로 쓰는 과정에서 future frame 또는 test video 정보가 누출되면 안 된다.

### 초기 Go/No-Go gate

- complete state transfer(feature + pointer + presence + metadata)를 구현한다.
- hard occlusion/reappearance subset에서 Last-Mask 대비 유의미한 이득이 있는지 본다.
- replay-k와 accuracy/latency Pareto frontier를 비교한다.
- Large→Small과 Small→Large를 분리해 원인을 분석한다.
- 이 단계에서 가치가 보이면 cross-architecture로 확장한다.

## 14. 프로젝트 범위

### 포함

- frozen pretrained stateful video models 사이의 runtime handoff
- memory/state translator
- cross-scale 및 cross-architecture transfer
- downstream-aware training
- replay 절감과 handoff 비용 평가
- 반복 switch와 실패 조건 분석

### 기본적으로 제외

- 새로운 VOS backbone 자체 개발
- 모델 선택 router를 논문의 주 기여로 개발
- 전체 end-to-end foundation model 재학습
- 단순 offline feature distillation만 수행하는 연구
- tensor MSE만 낮추고 후속 비디오 성능을 평가하지 않는 연구

## 15. 논문 포지셔닝 초안

> We study runtime handoff between heterogeneous stateful video models. Instead of replaying the entire video prefix after a model switch, a lightweight translator converts the source model’s accumulated spatial and object-centric temporal memory into a target-compatible state. The target resumes inference from the next frame. We evaluate the handoff by post-switch segmentation, identity continuity, occlusion recovery, repeated-switch stability, and end-to-end switching cost, against reset, last-mask, replay, target-native, and dynamic-submodel baselines.

과장 없이 주장할 수 있는 잠재적 기여:

1. heterogeneous VOS model 사이의 stateful handoff 문제를 명확히 정의
2. spatial memory, object pointer, presence/metadata를 포함한 component-aware translator
3. post-switch downstream behavior 중심의 학습·평가 방법
4. replay와 Last-Mask를 중심으로 한 accuracy–latency 분석 및 RDS와의 문제 범위 구분
5. transferability, 방향 비대칭, 반복 전환 drift, failure boundary에 대한 실증적 연구

## 16. 실행 순서

1. SAM 2 코드에서 실제 inference state schema를 정확히 추출하고 저장/주입 round-trip test를 만든다.
2. 동일 checkpoint의 controlled state perturbation으로 어떤 component가 중요한지 ablation한다.
3. SAM 2 small↔large에서 linear/MLP translator와 complete-state handoff를 구현한다.
4. hard-event benchmark와 필수 baseline을 먼저 구축한다.
5. downstream/multi-step loss와 hybrid replay를 평가한다.
6. XMem 또는 Cutie를 추가해 cross-architecture pair를 만든다.
7. edge↔cloud handoff 시나리오를 정리하고, 동일 조건이 성립할 때만 RDS와 matched-budget 비교를 수행한다.
8. 반복 switch, bandwidth, drift, uncertainty-triggered refresh를 분석한다.

## 17. 결정 기록

### 2026-08-20 — 프로젝트 메모리의 저장 위치

- 결정: 프로젝트의 새 채팅이 자동으로 기초 지식을 읽도록 루트 `AGENTS.md`와 `PROJECT_CONTEXT.md`를 사용한다.
- 이유: 노션에만 기록하면 새 채팅이 자동으로 프로젝트 규칙과 문맥을 가져간다는 보장이 없기 때문이다.
- 운영: 노션은 연구 원자료로 유지하고, 로컬 파일은 검증·요약된 시작 컨텍스트로 유지한다.
- 중요한 제한: 사용자가 명시하지 않으면 노션을 수정하지 않는다.

### 2026-08-20 — 핵심 연구 정의

- 결정: 프로젝트의 핵심은 “small memory를 large memory와 똑같이 복원”하는 데 있지 않고, heterogeneous target이 replay 없이 유용하게 이어서 동작하도록 하는 runtime state handoff다.
- 성공 기준: downstream continuity와 실제 시스템 비용을 함께 만족해야 한다.

### 2026-08-20 — RDS 대응

- 당시 결정: RDS를 핵심 경쟁 대안으로 취급했다.
- 변경: 아래 `RDS 우선순위 재분류` 결정에 따라 직접적인 method baseline이 아니라 scenario-level 대안으로 낮췄다.

### 2026-08-20 — RDS 우선순위 재분류

- 결정: RDS는 각 프레임의 연산량을 단일 모델 내부에서 동적으로 조절하며, 서로 다른 모델의 누적 memory를 호환 가능한 state로 번역하는 연구가 아니다.
- 운영: translator 설계를 위한 핵심 방법론 목록에서는 제외하고, 실제 heterogeneous/cross-device model switch가 왜 필요한지 설명하는 related-work 및 문제 설정 방어 자료로만 사용한다.
- 실험: CMMT와 동일한 사용 시나리오와 예산을 공정하게 구성할 수 있을 때만 RDS를 empirical baseline으로 둔다.

### 2026-08-20 — SAM 2.1 same-family 실험의 역할

- 확인: SAM 2.1 Tiny/Small/Base+/Large는 서로 다른 backbone 규모에도 공통된 256-d memory-attention 경계, 64-d memory-encoder 출력, 7개 memory slot 및 object-pointer 계열 인터페이스를 사용한다.
- 결정: 첫 controlled pilot은 Tiny↔Large로 하며, LLM cross-model KV transfer에서처럼 direct copy → component-wise ridge/linear map → 작은 nonlinear mapper 순서로 난도를 높인다.
- 판단 기준: direct copy나 Last-Mask가 이미 oracle에 가까우면 same-family learned translator만으로는 논문 기여가 약하다. learned mapping이 downstream continuity를 유의하게 개선하고 replay보다 싸야 의미가 있다.
- 포지셔닝: SAM 2를 2026년의 보편적 SOTA라고 주장하지 않는다. SAM 3와 SAM2Long 등 후속·특화 모델이 존재하므로, SAM 2의 가치는 공개된 stateful video foundation model, 명시적 streaming memory, 규모가 다른 동일-family checkpoints를 제공하는 재현 가능한 testbed라는 데 둔다.
- 확장: 최종 일반성은 cross-architecture pair 또는 여러 family/model pair에서 입증한다.

### 2026-08-22 — C 담당 Translator 방법론 기본안

- 근거 문서: 로컬 `docs/design/C_TRANSLATOR_METHOD_REPORT.md`; SAM 2 공식 논문·공식 `SAM2VideoPredictor`/`SAM2Base` 코드와 SAM 2.1 Tiny/Small/Large config를 재확인했다.
- **[확인]** 공식 predictor의 frame별 compact output에는 `maskmem_features`, `maskmem_pos_enc`, `pred_masks`, `obj_ptr`, `object_score_logits`가 들어간다. 실제 continuation state는 여기에 per-object conditioning/non-conditioning history, frame/object mapping, tracking metadata를 포함한다.
- 결정: translator 학습 전에 동일 checkpoint의 complete-state export→inject round-trip을 통과시킨다. Portable state는 현재 attention tensor만이 아니라 이후 target slot selection에도 충분한 **continuation-closed state**여야 하며, injection 뒤 target의 과거 frame encoder 호출은 0회여야 한다.
- 결정: 첫 cross-model baseline은 target positional information을 재생성하는 Direct Copy다. `maskmem_pos_enc`, spatial/temporal position과 pointer temporal encoding은 target-side에서 regenerate하고, frame index·conditioning flag·slot order·validity·external object ID는 learned translation 없이 보존·검증한다.
- 결정: 첫 learned MVP는 direction-specific component-wise residual Linear/Ridge다. Spatial memory와 `obj_ptr`는 별도 head, presence는 scalar calibration과 feature/pointer conditioning에 사용한다. 예상 SAM 2.1 contract에서 약 70K parameters/direction이며 실제 shape 확인 뒤 자동 산출한다.
- 결정: 다음 구조는 residual 2-layer MLP다. Attention/Transformer는 round-trip/metadata 오류를 배제하고 MLP가 Linear보다 유의하게 개선하면서도 slot/context interaction residual이 정량적으로 남을 때만 slot-summary factorized attention으로 도입한다. 모든 spatial token에 global attention을 거는 구조는 기본안이 아니다.
- 결정: 학습은 paired-state reconstruction → one-step target readout/mask distillation → drift가 확인될 때만 3–5 step rollout 순서다. Source/target parameter는 freeze하고 student target 연산을 통해 translator에만 gradient를 보낸다.
- 결정: component ablation은 feature-only → feature+pointer → full presence-aware → downstream loss → rollout의 nested 비교를 사용한다. Hybrid는 translation+replay-1/2/4까지만 먼저 평가하고 Pareto 개선이 이어질 때만 더 긴 replay로 확장한다.
- **[설계·미검증]** 초기 판정 기준은 same-checkpoint round-trip J&F 차이 0.1 point 이하, Direct Copy가 oracle 1 point 이내이면 same-family learned translator 생략, learned translator가 hard-event에서 Last-Mask 대비 2 points 이상 및 clustered CI 우위를 보이면서 replay frontier에 Pareto-dominated되지 않는 것이다. Pilot 분산을 본 뒤 test 전에 한 번만 고정한다.
- 역할 충돌 해소: 첫 controlled pair는 최신 결정대로 Tiny↔Large를 사용하며, `StateSpec`에 channel/grid/slot contract를 넣어 역할 분담표의 Small↔Large도 config 교체로 실행 가능하게 한다.

### 2026-08-22 — C 역할 2인 분담

- 결정: C 담당 2명은 Tiny→Large/Large→Tiny처럼 방향별로 나누지 않는다. 방향별 분담은 공통 code path를 중복시키고 방향 비대칭과 구현 차이를 혼동시키기 때문이다.
- **C1 — Translator Core & Offline Alignment:** canonical state/schema validator, Direct Copy와 round-trip, paired-state offline pipeline와 train-only statistics, Ridge/Linear/MLP, 조건부 resampler/attention, translator complexity/latency, D에게 넘길 architecture-independent interface를 담당한다.
- **C2 — Downstream Training, Ablation & Systems Evaluation:** component/readout/mask/rollout loss, Stage 1/2/3 trainer, B evaluator integration, C0–C10 ablation, translation+short replay, identity/recovery/repeated-switch, end-to-end accuracy–latency Pareto와 Go/No-Go 문서를 담당한다.
- 경계: C1은 A의 raw extraction/injection을 대신 구현하지 않고 contract를 소비·검증한다. C2는 B의 baseline/metric을 포크하지 않고 같은 evaluator를 소비한다.
- 공동 gate: round-trip 승인 → Linear/Ridge MVP freeze → C1–C4 완료 → MLP/hybrid/attention escalation review → validation config 공동 서명 → test 1회 순서로 운영한다.
- 상세 task/주간 일정은 `docs/design/C_TRANSLATOR_METHOD_REPORT.md` §8.6과 §9.4에 기록했다.

### 2026-08-25 — 로컬 저장소 구조와 Git 원격

- 결정: 프로젝트 진입 문서인 `AGENTS.md`, `PROJECT_CONTEXT.md`, `README.md`는 루트에 유지하고, 설계 문서는 `docs/design/`, 논문과 텍스트 추출본은 `references/`, 구현 package는 `src/`, 검증은 `tests/`, 실행 설정과 진입점은 각각 `configs/`, `scripts/`로 분리한다. 현재 구현 package는 `src/vos_memory_inspector/`다.
- 결정: 데이터셋, 모델 weight, checkpoint와 대용량 실험 산출물은 기본적으로 Git에서 제외하고, 재현에 필요한 코드·작은 설정·요약 결과만 추적한다.
- 당시 Git: 최초에는 ref가 없는 빈 `https://github.com/memorybridge-team/vos_menory_translater_KIM.git`을 `origin`으로 연결했다.
- 변경: 사용자 요청에 따라 위 연결을 제거하고 `origin`을 `https://github.com/memorybridge-team/vos-memory-translator.git`로 교체했다. 새 원격의 기본 브랜치는 `feature/sam2-memory-inspection`이며, 작업 브랜치는 처음 `kim/cmmt-development`로 만들었다가 2026-08-26 원격 이름 변경에 맞춰 `kim/exp-sam2-state-translator`로 변경했다.
- 이유: 연구 원자료, 검증된 설계, 실행 코드와 생성물을 구분해 탐색성을 높이고, 향후 코드 변경을 독립적인 commit으로 관리하기 위해서다.

### 2026-08-25 — Cross-Model KV 분석과 translator baseline 구현

- 근거 문서: `docs/design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md`; Heo et al. *Cross-Model KV Cache Transfer in LLM Families* arXiv:2608.03893v1, Hugging Face cache 문서, pinned SAM 2 commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`를 재확인했다.
- **[확인]** 논문의 production linear mapper는 target `(layer, KV head, K/V)`마다 독립적인 centered affine ridge(`lambda=0.01`)다. Target layer별 top-k source layers를 선택하고 각 선택 layer의 모든 source KV heads를 `[B*T, k*Hkv*Dh]` feature로 concatenate한다. Source RoPE를 inverse한 content key를 mapping한 뒤 target RoPE를 다시 적용한다.
- **[확인]** 논문 평가 pair는 모두 GQA `8 KV heads`, `head_dim=128`로 source/target cache shape가 맞는다. Head/layer/dimension mismatch를 수식상 처리할 수 있다는 것과 실제 검증됐다는 것은 구분한다. Matched shape에서도 Ministral 3→14/8→14는 낮은 retention을 보여 shape equality가 semantic compatibility를 보장하지 않는다.
- **[확인]** 논문 nonlinear mapper는 per target `(layer,head,K/V)` `Linear(ds,1024)→ReLU→Linear(1024,1024)→ReLU→Linear(1024,Dt)`, Adam `1e-3`, 20 epochs, MSE, batch 4096이다. Ridge 성공 pair에서는 보편적 개선이 아니며 일부 failure pair에서만 큰 개선을 보였다.
- 공개 코드 상태: 2026-08-25 기준 arXiv/논문에서 저자 공식 repository를 확인하지 못했다. `souvikDevloper/kvbridge@949d81d...`는 독립 구현 근거로만 사용하며 공식 코드로 인용하지 않는다.
- **[구현]** 기존 `src/vos_memory_inspector/`에 nested tensor inspector, Hugging Face legacy/current cache normalization, runtime-validated `CanonicalState`/`StateSpec`, SAM 2 multi-object cond/non-cond history canonicalizer, target positional factory가 필수인 history materializer를 추가했다.
- **[구현]** translator ladder는 Direct Copy, centered OLS/Ridge, component-wise Linear, separate feature/pointer residual two-layer MLP와 scalar presence calibrator다. Grid mismatch는 bilinear resampling, channel/pointer mismatch는 explicit projection 또는 Direct의 zero-pad/truncate로 처리하고 residual identity는 input/output dimension이 같을 때만 쓴다. Global spatial attention은 추가하지 않았다.
- **[Pilot — synthetic only]** Windows/Python 3.13/PyTorch 2.13 CPU에서 unit test `12 passed`; generated affine+quadratic paired state에서 Direct aggregate MSE 1.44755, Ridge 0.030507, Linear 0.111864, residual MLP 0.019728이었다. 이는 schema/fitting/metric/serialization smoke이며 SAM 2 J&F나 real handoff 성능이 아니다.
- **[미검증]** 로컬에 official SAM 2 checkout과 Tiny/Large checkpoint가 없어 paired runtime dump, same-checkpoint round-trip, target PE 재생성을 포함한 next-frame injection, DAVIS/J&F/identity/recovery/latency 측정은 실행하지 않았다. Checkpoint와 dataset은 자동 다운로드하지 않았다.

## 18. 노션 원자료 인덱스

프로젝트 데이터베이스:

- [CMMT 노션 데이터베이스](https://app.notion.com/p/3b9cace0ea5c80149ca5df8b9dec27f0?v=3b9cace0ea5c802092d7000c0f24221a)

페이지:

1. [읽어봐야할 논문인가요](https://app.notion.com/3b9cace0ea5c808a8209e8510f842b57) — 관련 연구 우선순위와 후보 목록
2. [아이디어](https://app.notion.com/3bacace0ea5c808582b8d50b88b4e1a5) — 핵심 발상 및 현재 정리
3. [GPT에게 연구해도 되는지 물어보았다](https://app.notion.com/3bacace0ea5c80a8b4aec2876ad056c2) — 초기 타당성 검토와 Pilot 기록
4. [SAM2](https://app.notion.com/3bacace0ea5c80b587b1eb11f755b2bf) — 초기 기반 모델 분석
5. [Cross-Model KV cache transfer](https://app.notion.com/3bacace0ea5c8041b5c5f7025d982ee0) — 가장 직접적인 translation 선행 연구
6. [Cache-to-Cache](https://app.notion.com/3bacace0ea5c80ef9762e8950f95125c) — source/target cache의 gated semantic fusion 참고
7. [Efficient VOS with RDS](https://app.notion.com/3bacace0ea5c80688673c4741d315ab7) — dynamic-submodel 경쟁 방식
8. [클라우드 GPU](https://app.notion.com/3c0cace0ea5c8035b566c88a25348280) — 실험 자원 후보 메모; 가격·가용성은 사용 전 재확인
9. [XMem](https://app.notion.com/3c1cace0ea5c80329da9c28f1693e0b6) — 현재 상세 분석 미작성
10. [SAM2Long](https://app.notion.com/3c1cace0ea5c80b09a5ecf397acdb579) — 현재 상세 분석 미작성
11. [Cross-Model Temporal Memory Translation for Stateful Video Inference](https://app.notion.com/3bacace0ea5c8024ba3dea08f1824d72) — 연구 제안서

## 19. 외부 핵심 출처

- [SAM 2 공식 저장소](https://github.com/facebookresearch/sam2)
- [SAM 2 공식 연구 페이지](https://ai.meta.com/research/sam2/)
- [SAM 2 논문](https://arxiv.org/abs/2408.00714)
- [SAM 3 공식 연구 페이지](https://ai.meta.com/research/sam3/)
- [SAM 2.1 Tiny 설정](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml)
- [SAM 2.1 Small 설정](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml)
- [SAM 2.1 Base+ 설정](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_b%2B.yaml)
- [SAM 2.1 Large 설정](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml)
- [RDS — CVPR 2026 Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Efficient_Video_Object_Segmentation_and_Tracking_with_Recurrent_Dynamic_Submodel_CVPR_2026_paper.html)
- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)
- [Cache-to-Cache](https://arxiv.org/abs/2510.03215)

## 20. 문헌 읽기 우선순위 — 2026-08-20 검토

### Tier A — 프로젝트 팀 전원이 반드시 읽기

- SAM 2: 실제 state schema와 streaming memory의 기준 모델
- XMem: sensory/working/long-term memory를 가진 대표적인 cross-architecture 후보
- Cutie: object-level memory와 distractor 대응을 이해하기 위한 핵심 VOS 모델
- Cross-Model KV Cache Transfer: replay/prefill을 피하는 cross-model state mapping의 가장 직접적인 방법론 선행연구
- Mixture-of-Translators: heterogeneous architecture 사이의 translator mixture, correction loss, injection-shift 분석
- Rethinking Memory Design in SAM-Based Visual Object Tracking: short-term appearance memory와 long-term distractor-resolving memory를 포함한 memory design의 체계적 비교

### Tier B — 담당자를 정해 정독하고 팀에 공유

- SAM2Long: memory error accumulation과 장기 영상 복구
- RMem: memory가 많을수록 항상 좋은 것이 아니며, 선택·용량·freshness가 중요하다는 근거
- Efficient Track Anything(EfficientTAM): lightweight target 및 edge↔cloud handoff 후보
- Latent Space Translation via Semantic Alignment 또는 최신 VFM Model Stitching: representation alignment의 대조 방법
- Cache-to-Cache: target cache와 source semantics를 융합하는 gated communication 참고
- MOSEv2와 LVOS benchmark: occlusion, reappearance, distractor, long-video 평가 조건 설계

### Tier C — 필요 섹션만 읽기

- RDS(CVPR 2026): translator 직접 선행연구가 아니라 단일 모델 dynamic-compute라는 scenario-level 대안. 문제 설정과 한계 부분 위주
- IAM, DroidSpeak: attention mapping과 selective recomputation baseline 아이디어
- Efficient-SAM2, TinySAM 2, EfficientSAM3: switching 없이 memory/연산을 압축하는 경쟁 대안
- ARTrack-AC: tracking의 adaptive-capacity 경쟁 방식
- SAMURAI, DAM4SAM, SENTRY: memory write/selection과 distractor 대응 세부 설계
- Stateful Worlds, Stateless Elasticity: same-model exact-state migration과 시스템 비용 논의
- DAVIS, YouTube-VOS: 전체 논문보다 metric/protocol 부분 위주

### 우선 제외 또는 대체

- Zero-shot stitching in Reinforcement Learning: 현재 VOS memory handoff와 거리가 멀어, `Latent Space Translation via Semantic Alignment` 또는 vision-foundation-model stitching으로 대체
- Phi-Mamba/MOHAWK: offline architecture distillation이므로 runtime state handoff 방법을 설계할 때 직접 필요하지 않음
- TMAM: SAM 2 memory mechanism을 2D surgical segmentation에 이식하지만 누적 runtime state를 다른 모델에 번역하지 않으므로 related-work 구분용
- STSeg: per-video model selection/pseudo-label refinement이며 state handoff가 없어 benchmark SOTA 확인 외에는 후순위
- 일반 MS/GEMEL/Ekya/Chameleon 계열: stateless model sharing/switching 배경용이며 개별 정독 불필요
- 경량 adaptive video inference(arXiv:2601.14568): 모델 switching 동기는 유사하지만 stateful VOS가 아니므로 RDS보다 우선하지 않음
- vec2vec: generic embedding translation으로, 실제 translator가 paired state mapping에서 막힐 때만 재검토

### 노션 상태 판단

- 노션의 `논문` 분류 6개 중 SAM 2, Cross-Model KV Cache Transfer, RDS, XMem, SAM2Long은 유지한다.
- Cache-to-Cache는 삭제하지 않되 핵심 필수에서 보조 방법론으로 낮춘다.
- XMem과 SAM2Long 페이지는 현재 템플릿만 있고 분석이 비어 있으므로 제거 대상이 아니라 보완 대상이다.
- Cutie, Mixture-of-Translators, Rethinking Memory Design, RMem, EfficientTAM, MOSEv2/LVOS는 별도 정리 페이지가 없거나 목록에만 있으므로 추가 분석 대상이다.

## 21. 동기화 체크리스트

새로운 중요한 결정이나 결과가 생기면 다음을 갱신한다.

- 문서 상단의 로컬 스냅샷 기준일
- 상태 표기 `[확인]`, `[가설]`, `[Pilot]`
- 성공 기준이나 scope가 변했는지
- 새 baseline/dataset/metric
- Pilot 또는 정식 실험 결과와 실험 조건
- 실패하거나 폐기한 접근과 이유
- 결정 기록
- 노션/논문/코드 출처 링크

노션에서 동기화할 때에는 로컬의 사용자 결정과 노션의 연구 노트를 구분해 병합하며, 기존 기록을 근거 없이 덮어쓰지 않는다.
