# Repository cleanup review — 2026-09-09

현재 정식 실험 계획, 재현성, 공개 Pages와 제3자 자료의 저작권 범위를 기준으로
작업 브랜치의 파일을 검토했다.

## Git 추적 해제

- `references/papers/*.pdf`: 공식 링크로 다시 받을 수 있는 제3자 논문 원문이며
  세 파일이 약 23.6MB를 차지한다.
- `references/extracted_text/`: 위 논문을 페이지별로 복제한 검색용 파생 파일이다.

두 경로는 로컬 연구 캐시로는 유지할 수 있지만 Git에는 필요하지 않다. `.gitignore`에
추가하고, `references/README.md`에는 공식 링크와 프로젝트 분석 문서만 남긴다.

## 유지

- `reports/experiments/`의 과거 smoke와 pilot: 성공 주장용이 아니라 실제 실행·실패
  조건을 추적하는 감사 기록이다.
- `docs/experiments/2026-09-08-davis-handoff/`: 다른 컴퓨터에서 결과를 확인하기 위한
  GitHub Pages의 직접 소스다.
- `PROJECT_CONTEXT.md`, run logs, JSON metrics: 실험 조건과 판단 변경을 보존한다.
- 작은 대표 mask/PNG: 보고서가 GitHub와 VS Code에서 자체적으로 열리게 한다.

## 통합하거나 표시를 수정

- 루트 `README.md`: 현재 목표, 실제 pilot 경계, 공개 결과, 다음 실험과 실행 방법을
  한국어로 다시 작성한다.
- `docs/runpod.md`: 과거 12달러/20시간 cap을 제거하고 연구 완전성 우선 정책으로 바꾼다.
- `docs/design/SAM2_TRANSLATOR_EXPERIMENT_PLAN.md`: 삭제하지 않고 초기 계획 보관본으로
  명시하며 현재 `docs/experimental_plan.md`로 연결한다.

## 계속 Git에서 제외

Checkpoint, dataset 원본, raw tensor/state, 대량 비공개 predictions, SSH key,
환경변수와 임시 폴더는 재현 명령과 manifest만 남기고 파일 자체는 올리지 않는다.
