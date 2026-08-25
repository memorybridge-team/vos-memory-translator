# Cross-Model Memory Translator — Project Instructions

이 디렉터리는 **Cross-Model Memory Translator(CMMT)** 연구 프로젝트 전용 작업공간이다.

## 새 대화 시작 시

- 이 프로젝트와 관련된 질문·분석·구현을 시작하기 전에 반드시 `PROJECT_CONTEXT.md`를 끝까지 읽는다.
- 사용자의 최신 지시가 이 파일이나 `PROJECT_CONTEXT.md`보다 우선한다.
- 기본 답변 언어는 한국어로 하되, 논문명·모듈명·평가지표 등은 통용되는 영문 표기를 함께 쓴다.
- 노션은 원자료와 진행 기록, `PROJECT_CONTEXT.md`는 새 대화를 위한 검증된 로컬 스냅샷으로 취급한다.
- 사용자가 최신 노션 내용을 확인해 달라고 하거나 로컬 스냅샷의 최신성이 중요한 경우 노션을 다시 읽는다.
- **사용자가 명시적으로 요청하지 않으면 노션 페이지를 수정하지 않는다.**

## 프로젝트의 고정된 핵심

- 문제는 장시간 비디오의 중간 시점에 상태를 가진 비디오 모델을 교체할 때, 새 모델이 과거 프레임을 다시 처리해야 하는 비용이다.
- 핵심 아이디어는 source model의 temporal memory/state를 작은 translator로 target model이 이어서 사용할 수 있는 state로 변환해, 과거 프레임 replay 없이 다음 프레임부터 계속 추론하게 하는 것이다.
- 기호는 source state `b_t`, target이 과거를 직접 처리해 얻은 native/oracle state `a_t`, 번역된 state `â_t = T(b_t)`를 기본으로 쓴다.
- 목표는 `â_t`의 텐서 복원 오차를 최소화하는 것 자체가 아니라, target model의 **후속 비디오 작업 성능·identity continuity·occlusion recovery를 유지하면서 handoff 비용을 줄이는 것**이다.
- 주요 초기 태스크는 Video Object Segmentation/Tracking(VOS/VOT)이며, SAM 2를 첫 구현 기반으로 삼되 최종 가치는 SAM 2 내부 small↔large를 넘어 SAM 2↔Cutie/XMem 같은 이질적 구조에서 입증해야 한다.
- 주 연구 범위는 모델 선택 router나 새 backbone 개발이 아니라 **state/memory handoff**다.

## 연구 판단 원칙

- 다음 세 종류를 섞지 않는다: 논문이나 코드로 확인된 사실, 아직 검증하지 않은 가설, 제한적인 pilot 결과.
- tensor-level reconstruction보다 downstream J&F, switch 직후 성능, identity break, recovery length, latency/FLOPs/VRAM을 우선한다.
- 직접 비교군인 target reset, Last-Mask, replay-k, direct transfer, target-native oracle를 우선한다. RDS 같은 dynamic-compute 방식은 동일한 모델 전환 문제를 푸는 방법이 아니라 모델 전환 자체를 피하는 scenario-level 대안이므로, 실험 조건이 맞을 때만 별도로 비교한다.
- 특히 Last-Mask가 강할 수 있으며, easy benchmark의 평균 J&F만으로 연구 가치를 주장하지 않는다.
- Small→Large는 작은 모델이 이미 버린 정보를 완벽히 복구할 수 없다는 한계를 전제로 한다. 필요하면 translated memory + short replay의 hybrid를 정당한 설계로 검토한다.
- “최초의 cross-model state transfer”처럼 넓은 선점 주장은 피한다. 핵심 novelty는 **runtime cross-architecture spatial/object-centric temporal-memory handoff without full replay**로 좁혀 표현한다.
- RDS(CVPR 2026)는 translator의 직접 선행연구가 아니다. related work에서 단일 모델 내부 dynamic submodel과 CMMT의 heterogeneous handoff 문제를 구분하고, 서로 다른 모델·구조·장치 간 실제 교체가 필요한 시나리오에서만 경쟁 대안으로 다룬다.

## 로컬 메모리 유지 규칙

- 대화에서 프로젝트의 목표, 범위, 방법, 실험 설계, 결정, 확인된 결과 또는 주요 위험이 실질적으로 바뀌면, 사용자가 금지하지 않는 한 답변을 마치기 전에 `PROJECT_CONTEXT.md`를 갱신한다.
- 새 내용에는 날짜와 근거를 남기고, 불확실한 내용은 `가설` 또는 `미검증`이라고 표시한다.
- 기존 결정과 출처를 조용히 지우지 않는다. 변경된 판단은 결정 기록에 이전 판단과 변경 이유를 남긴다.
- `AGENTS.md`는 짧고 안정적인 행동 규칙만 유지하고, 상세 연구 내용은 `PROJECT_CONTEXT.md`에 둔다.
