# Cross-Model Memory Translator — Project Instructions

이 디렉터리는 **Cross-Model Memory Translator(CMMT)** 연구 프로젝트 전용 작업공간이다.

## 새 대화 시작 시

- 이 프로젝트와 관련된 질문·분석·구현을 시작하기 전에 반드시 `PROJECT_CONTEXT.md`를 끝까지 읽는다.
- 사용자의 최신 지시가 이 파일이나 `PROJECT_CONTEXT.md`보다 우선한다.
- 기본 답변 언어는 한국어로 하되, 논문명·모듈명·평가지표 등은 통용되는 영문 표기를 함께 쓴다.
- 노션은 원자료와 진행 기록, `PROJECT_CONTEXT.md`는 새 대화를 위한 검증된 로컬 스냅샷으로 취급한다.
- 사용자가 최신 노션 내용을 확인해 달라고 하거나 로컬 스냅샷의 최신성이 중요한 경우 노션을 다시 읽는다.
- **사용자가 명시적으로 요청하지 않으면 노션 페이지를 수정하지 않는다.**

## 업무와 결과물의 기본 운영

- 답변·분석만 요청받은 경우에는 근거를 확인해 보고하되, 외부 서비스나 저장소를 임의로 수정하지 않는다. 구현·문서 작성·수정이 요청되면 필요한 범위까지 실제로 완성하고 위험에 맞게 검증한다.
- 작업 시작 시 기대 결과와 검증 기준을 먼저 정한다. 장시간·다단계 작업은 확인 가능한 산출물 단위로 나누되, GitHub Issue·담당자·검토자·기한은 팀이 실제로 사용하는 작업이거나 사용자가 요청한 경우에만 생성·지정한다.
- 코드는 실행 방법, 설정·seed, 테스트 또는 검증 결과, 생성물 위치와 알려진 한계를 함께 남긴다. 연구 조사는 출처, 확인된 사실, 가설, 프로젝트에 미치는 판단을 구분한다.
- 기존 사용자 변경을 보존하고, `main` 직접 push나 외부 게시·메시지·PR 생성은 사용자의 요청 또는 이미 정해진 팀 절차가 있을 때만 수행한다.
- 진행 보고의 고정 시각, 주간 업무시간, 회의 요일, 응답 SLA는 자동 강제하지 않는다. 팀이 별도로 합의한 일정이 있을 때만 적용하며, blocker와 일정 영향은 확인되는 즉시 공유한다.
- 필요한 도구나 연결된 플러그인은 작업 범위 안에서 활용한다. 새 설치·계정 연결·추가 권한이 필요한 경우에는 제품의 승인 절차를 따르며 이를 우회하거나 대신 승인하지 않는다.
- 상세 운영 가이드의 원본은 Notion `AI 연구 프로젝트 협업 운영 가이드`이며, 이 파일에는 새 대화에도 반드시 적용할 안정 규칙만 유지한다.

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

## 회의록 운영

- 사용자가 `meetings/`에 회의 원문 `.txt`를 제공하면, 잡담과 프로젝트 무관 대화는 제외하고 같은 폴더에 `<원문명>_회의록.md`를 작성한다.
- 회의록은 회의 정보, 핵심 논의, 결정사항, 액션 아이템, 미해결 쟁점 순으로 정리하되 회의 정보에서 참석자와 원문 항목은 제외한다.
- 액션 아이템은 같은 담당자의 할 일을 하나로 묶고 기한 항목은 작성하지 않는다.
- 원문에 없는 결정·담당자를 추측하지 않으며, 불명확한 내용은 `미확정`으로 표시한다.
- Markdown 회의록을 작성한 뒤 Notion `회의 보고서` 데이터베이스(`https://app.notion.com/p/3c0cace0ea5c800a9f25defab29d531a?v=3c0cace0ea5c80ccafa8000ca380324d`)에 자동 업로드한다. 이 항목은 해당 회의록 업로드에 대한 사용자의 지속적 명시 권한이며 다른 Notion 페이지 수정 권한으로 확장하지 않는다.
- Notion의 `Doc name`은 회의 날짜의 `YYYY-MM-DD` 형식으로 지정한다. 같은 날짜 페이지가 이미 있으면 중복 생성하지 않고 해당 페이지를 최신 회의록 내용으로 갱신하며 `참석자` 속성은 비워 둔다.
- 회의에서 프로젝트의 목표·범위·방법·실험 설계·결정·결과·위험이 실질적으로 바뀌면 기존 로컬 메모리 유지 규칙에 따라 `PROJECT_CONTEXT.md`도 갱신한다.
