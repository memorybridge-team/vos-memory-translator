# L4 prepared-case cache smoke

## 목적

동일한 DAVIS case에서 source Tiny prefix와 target Large-native oracle을 한 번만
계산한 뒤, Direct candidate가 checksummed cache를 재사용해 과거 계산 없이 같은
future output을 만드는지 검증했다.

## 환경

- GPU: NVIDIA L4, 23,034 MiB reported VRAM
- Container disk: 20GB; current video와 Tiny/Large checkpoint 1,018MB를 local
  `/tmp/cmmt-hot-92629a9`에 stage
- Network Volume: 200GB, `/workspace`
- CMMT commit: `92629a90123eae5e126ccd52276285ac1f28b4fd`
- SAM 2 commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- Dataset: DAVIS 2017 val `bike-packing`, object 1, switch frame 14, seed 7

## 결과

| 항목 | 결과 |
|---|---:|
| 공통 준비 | Tiny prefix 15 frames + Large oracle future 54 frames |
| 공통 준비 시간 | 45.2708 s |
| 공통 준비 peak VRAM | 1,660,070,400 B |
| Cache 크기 | 184,075,799 B |
| Cache SHA-256 | `2d0a8b88d47095f28ca25bb156e326ab98a46c1cd51baff83894721f835663cd` |
| Cached Direct candidate 시간 | 30.6552 s |
| 기존 전체 Direct 시간 | 73.2373 s |
| 단일 candidate 구간 절감 | 58.14% / 2.39× |
| 과거 backbone 호출 | 주입 전 0, 주입 중 0 |
| Future backbone 호출 | 54 |
| Large-native agreement IoU | 0.0 |
| Large-native logit MSE | 94.9600334 |

Cached와 legacy 실행의 54개 frame별 결과는 정확히 같았다. 따라서 cache 경로는
실험 결과를 바꾸지 않으면서 두 번째 방법부터 source prefix와 oracle 재계산을
제거한다. Direct의 native-agreement IoU 0은 이 case에서 Direct Transfer가 실패한
결과이지 cache 실패가 아니다. DAVIS GT J&F는 아직 이 smoke에 포함하지 않았다.

6개 candidate를 비교하면 기존 방식의 단순 예상은 약 `6×73.24=439초`, cache
방식은 `45.27+6×30.66=229초`로 약 48% 줄어든다. 실제 각 baseline의 replay 비용이
다르므로 이 수치는 Direct와 같은 continuation 비용을 가정한 처리량 추정이다.

## 산출물

- `bike-packing_obj1_switch14.prepare.json`: 공통 reference/cache 생성
- `bike-packing_obj1_switch14.direct.json`: cached Direct
- `bike-packing_obj1_switch14.legacy-direct.json`: 기존 전체 Direct 대조

Raw `.pt` cache, DAVIS 원본과 checkpoint는 Network Volume에만 보존하고 Git에는
올리지 않는다.
