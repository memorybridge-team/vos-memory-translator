# CMMT 저장·입출력 파이프라인

## 결정

GPU가 학습 중 네트워크 파일을 기다리지 않도록 **hot/warm/cold 3단계**로 운영한다.
현재 DAVIS와 SAM 2 checkpoint는 합계 약 2.1GB로 작고 RunPod volume disk에 이미
검증되어 있으므로, 외부 저장소로 옮긴 뒤 매 frame 원격 접근하는 방식은 쓰지 않는다.

```text
GitHub (코드·설정)
        │ pull
        ▼
RunPod persistent /workspace ── stage current case ── local container hot cache
        │                                                   │
        │                                                   ▼
        │                                      GPU inference/training
        │                                                   │
        ◀──────────── atomic completed bundle ──────────────┘
        │
        └── 비동기/실험 종료 후 ──▶ 외부 cold object storage
```

## 계층별 역할

### Hot — RunPod local storage

- 현재 실험의 image sequence와 annotation
- 현재 source/target checkpoint
- paired canonical state와 Translator checkpoint
- 실행 중 prediction과 중간 checkpoint

Volume disk를 쓰는 단일 Pod에서는 `/workspace/CMMT` 자체가 local hot storage다.
Network Volume을 쓰는 이동형 구성에서는 현재 case만 container disk의 임시 경로로
stage한다. GPU가 읽는 파일은 모두 실험 시작 전에 이 계층에 있어야 하며, object
storage나 lazy mount에서 수천 개 JPEG/PNG를 학습 loop 중 직접 읽지 않는다.

### Persistent — RunPod `/workspace`

- 기본 구성은 Pod와 독립적인 500GB Network Volume
- Community Cloud 대안은 Pod 생명주기에 묶인 500GB Volume disk
- dataset, checkpoint, checksummed case cache와 완료 bundle의 source of truth

Network Volume은 volume disk를 대체한다. 둘을 동시에 요구하지 않는다.

### Warm — GitHub

- 코드, 고정 config와 case manifest
- dataset/checkpoint의 URL·revision·checksum
- 작은 metric JSON, 보고서와 선별한 시각화

Dataset 절대 경로를 manifest에 넣지 않는다. 실행할 때 `--root`로 주입해 같은
manifest를 다른 Pod와 로컬 검증에서 재사용한다.

### Cold — 외부 object storage

- 보존할 raw paired state
- 대규모 prediction/mask bundle
- 학습 checkpoint와 재개 지점
- 압축한 dataset shard 또는 공식 원본의 비공개 cache

외부 backend는 S3/GCS/Azure/B2 중 하나를 정한 뒤 연결한다. 인증정보는 최소 권한
secret으로만 주입하며 Git, report와 shell history에 기록하지 않는다.

## 실행 순서

1. GitHub branch를 fast-forward하고 test를 통과시킨다.
2. case manifest checksum과 필요한 hot file의 존재·여유 공간을 확인한다.
3. 누락된 sequence/checkpoint만 원격에서 `.partial` 파일로 내려받는다.
4. checksum이 맞으면 atomic rename한 뒤 GPU 작업을 시작한다.
5. GPU 결과는 먼저 local run directory에 완결된 bundle로 저장한다.
6. DAVIS metric과 report 생성을 CPU에서 수행한다.
7. GPU critical path가 끝난 뒤 cold storage 업로드를 수행한다.
8. 원격 object의 size/checksum을 검증한 뒤에만 local cold artifact를 정리한다.

업로드 실패는 학습 실패로 만들지 않는다. 완료 bundle을 hot storage에 남기고
업로드만 재시도한다. 반대로 필수 input download/checksum 실패 시에는 GPU를
시작하지 않는 fail-closed 정책을 사용한다.

## 현재 상태

- 현재 RunPod hot working set은 약 3.8GB이며 필수 자산 누락이 없다.
- 다음 Pod 권장값은 50GB container disk와 500GB Network Volume이다. Community
  Cloud를 선택할 때에는 Network Volume 대신 500GB Volume disk를 사용한다.
- 외부 cold storage backend와 credential은 아직 설정하지 않았다.
- 따라서 현재 정식 baseline은 기존 hot data로 실행하며 네트워크 I/O를 critical
  path에 추가하지 않는다.
- output이 volume 운영 한도에 접근하거나 MOSE/LVOS를 도입하기 전에 cold backend를
  연결하고, sequence별 tar/WebDataset shard의 stage-throughput을 측정한다.
