# 문서 인덱스

## Verified implementation documents

- [`memory_tensor_inventory.md`](memory_tensor_inventory.md): pinned SAM 2 upstream에서 확인한 temporal-memory producer, storage, consumer 경로
- [`validation.md`](validation.md): 현재 memory-inspection milestone의 실행·검증 기록

## Design

- [`design/C_TRANSLATOR_METHOD_REPORT.md`](design/C_TRANSLATOR_METHOD_REPORT.md): SAM 2.1 Tiny↔Large 기반 translator 구조, state contract, 학습 단계, ablation과 역할 분담 설계
- [`design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md`](design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md): Cross-Model KV Cache Transfer의 정확한 tensor/mapping 분석, SAM 2 adaptation, 구현 파일, synthetic 실행 결과와 실제 checkpoint blocker

새 설계 문서는 주제별 Markdown 파일로 `docs/design/`에 추가합니다. 실험 결과는 코드·설정·원시 metric과 연결되는 형태로 기록하고, 연구의 확정된 결정은 루트 `PROJECT_CONTEXT.md`에도 반영합니다.
