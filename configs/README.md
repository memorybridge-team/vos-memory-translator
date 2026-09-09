# Configs

모델 쌍, 전환 방향, 데이터셋 split, switch scenario, translator 구조와 학습·평가 설정을 재현 가능한 파일로 관리합니다. 개인 경로나 credential은 `*.local.yaml`/`*.local.yml`에 두며 Git에는 포함하지 않습니다.

- `davis2017_phase1_manifest_policy.json`: DAVIS Phase 1 manifest의 일반
  quantile, 최소 prefix/future 길이와 GT event tag 기준입니다. 실제 case 목록에는
  절대 dataset 경로를 저장하지 않습니다.
