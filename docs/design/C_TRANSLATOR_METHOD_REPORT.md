# CMMT C 담당 — Translator·방법론 설계 보고서

> 작성일: 2026-08-22 (KST)  
> 적용 범위: frozen stateful VOS models 사이의 runtime state handoff  
> 1차 controlled pair: **SAM 2.1 Tiny↔Large**  
> 상태 표기: **[확인]** 공식 논문·공식 코드로 확인 / **[설계]** 본 보고서의 권고 / **[가설]** 실험 전 / **[확인 필요]** A 담당자의 runtime trace 필요

## 결론부터

- **가장 먼저 할 일**은 translator 학습이 아니라 동일 checkpoint의 complete-state export→inject round-trip이다. 이 테스트가 실패하면 이후 reconstruction·downstream 결과는 해석할 수 없다.
- **첫 cross-model baseline**은 target positional information을 재생성하는 `DirectCopy`다.
- **첫 learned MVP**는 direction-specific, component-wise residual Linear/Ridge다. spatial feature에는 slot·pixel 공유 1×1 channel map, `obj_ptr`에는 별도 linear map, presence에는 scalar calibrator를 사용한다. Presence를 F/p 입력에도 넣는 권장형은 SAM 2.1의 예상 contract에서 약 **70,274 parameters/direction**이다(순수 component map은 69,954).
- **MVP의 바로 다음 구조**는 component-wise residual 2-layer MLP다. Linear/MLP가 context-free mapping의 한계를 보였다는 증거 없이 attention을 넣지 않는다.
- `maskmem_pos_enc`, learned temporal position embedding, object-pointer temporal encoding은 source tensor를 번역하지 않고 **target model의 parameter와 canonical frame age로 regenerate**한다.
- `frame_idx`, conditioning 여부, object ID, slot validity/order는 학습 대상이 아니라 **보존·검증할 control metadata**다.
- `object_score_logits`는 compact state에 저장되지만, 공식 read path에서는 과거 spatial memory와 pointer에 occlusion 정보가 이미 반영되어 있다. 따라서 presence head를 별도 출력만 하고 끝내지 말고 feature/pointer head의 conditioning에도 사용한다.
- translation-only가 short replay에 계속 지면 `translation + replay-{1,2,4}`를 정식 방법으로 올린다. `replay-8/16` hybrid는 1/2/4에서 Pareto frontier가 계속 개선될 때만 확장한다.

역할 분담표의 Small↔Large와 최신 프로젝트 결정의 Tiny↔Large가 충돌한다. 최신 결정에 따라 **Tiny↔Large를 첫 pair**로 사용하되 모든 코드가 `StateSpec`으로 channel/grid/slot 수를 받게 하여 Small↔Large는 config 교체만으로 실행되게 한다.

---

## 0. 공식 구현에서 확인된 state 경계

### 0.1 확인된 사실

SAM 2 논문은 memory attention이 현재 frame embedding을 과거 prompted/unprompted spatial memories 및 object pointers에 cross-attend시키며, memory bank가 최근 frame과 prompted frame의 spatial feature map, high-level semantic vector인 object pointer를 보존한다고 설명한다. Presence head는 현재 frame에서 object가 존재하는지도 예측한다. 이는 [SAM 2 논문 §4](https://arxiv.org/html/2408.00714#S4)의 공식 설명이다.

공식 predictor의 frame별 compact output에는 다음 다섯 항목이 들어간다.

```text
maskmem_features
maskmem_pos_enc
pred_masks
obj_ptr
object_score_logits
```

근거는 [`SAM2VideoPredictor._run_single_frame_inference`](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py#L688-L752)다. predictor 전체 state에는 이것 외에 object ID↔index mapping, per-object conditioning/non-conditioning output dictionary, prompt input, tracked-frame metadata가 있다. 공식 초기화 구조는 [`SAM2VideoPredictor.init_state`](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py#L37-L94)에서 확인할 수 있다.

Memory read 시에는:

- target policy가 가까운 conditioning frames를 선택한다.
- 최근 `num_maskmem - 1` non-conditioning frames를 stride 정책에 따라 선택한다.
- `maskmem_features`와 spatial PE를 펼쳐 memory attention에 넣는다.
- temporal PE는 저장 tensor에서 가져오는 것이 아니라 target의 `maskmem_tpos_enc`를 slot age에 따라 더한다.
- object pointer도 frame distance로 temporal PE를 다시 만든다.

이 동작은 [`SAM2Base._prepare_memory_conditioned_features`](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py#L467-L640)에 명시되어 있다. 따라서 CMMT의 portable state는 “현재 선택된 tensor 묶음”만이 아니라, **다음 frame들에서도 target의 selection policy가 필요한 history를 찾을 수 있는 continuation-closed state**여야 한다.

`maskmem_pos_enc`는 공식 predictor에서 frame/object 간 동일한 값으로 취급되어 session constant로 한 번만 보관된다([공식 코드](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py#L787-L810)). 반면 temporal PE는 target model parameter다([공식 코드](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py#L116-L134)). 이 때문에 target-side regeneration이 기본안이다.

SAM 2.1 Tiny, Small, Large 공식 config는 모두 image-neck/memory-attention `d_model=256`, memory encoder `out_dim=64`, `num_maskmem=7`, memory-attention feature grid 64×64를 사용한다. Tiny와 Large의 backbone 규모는 다르지만 memory boundary config는 정렬되어 있다: [Tiny config](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml), [Small config](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml), [Large config](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml).

### 0.2 예상 runtime contract — A 담당자 확인 전에는 확정값이 아님

| Component | 공식 config/code로부터의 예상 | 상태 | A가 확인할 것 |
|---|---:|---|---|
| active `maskmem_features` | per object/slot `[1, 64, 64, 64]` | **[확인 필요]** | 실제 axis, batch/object packing, dtype, contiguity |
| `maskmem_pos_enc[-1]` | `[1, 64, 64, 64]`, session constant | **[확인 필요]** | list 길이, checkpoint 간 numerical identity |
| `obj_ptr` | per object/frame `[1, 256]` | **[확인 필요]** | exact dtype, pointer packing/splitting |
| `object_score_logits` | per object/frame `[1, 1]` 예상 | **[확인 필요]** | exact shape and read/re-encode paths |
| `pred_masks` | `[1, 1, 256, 256]` 예상 (`image_size/4`) | **[확인 필요]** | continuation에 필요한 범위, dtype |
| slot policy | cond outputs + recent non-cond, `num_maskmem=7` | **[확인]** | export해야 하는 최소 history closure |
| temporal position | stored tensor가 아니라 target parameter + frame distance | **[확인]** | injection 후 frame index가 동일하게 해석되는지 |

여기서 shape는 예상치다. 보고서의 parameter 수 예시는 이 예상치를 사용하지만, 구현 상수는 사용하지 않는다. 첫 schema dump가 다른 값을 보이면 `StateSpec`만 바꾸고 식을 다시 계산한다.

### 0.3 presence/occlusion의 주의점

공식 코드에서 no-object 정보는 두 곳에 이미 주입된다.

- spatial memory에는 object가 보이지 않을 때 `no_obj_embed_spatial`이 더해진다.
- `obj_ptr`에는 presence decision에 따라 `no_obj_ptr`이 혼합된다.

근거는 [`_encode_new_memory`](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py#L642-L689)와 [`_forward_sam_heads`](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py#L369-L379)다. 즉, 저장된 `object_score_logits`만 잘 번역해도 과거 slot의 feature/pointer가 target no-object semantics로 자동 교정되는 것은 아니다. 이 때문에 presence는 별도 scalar head이면서 feature/pointer translator의 입력 conditioning이어야 한다.

---

## 1. 문제 정식화와 설계 원칙

### 1.1 입력, 출력, target state contract

Object `o`, memory record/slot `k`에 대해 source state를 다음처럼 둔다.

\[
b_t = \{F^S_{ok}, p^S_{ok}, z^S_{ok}, m^S_{ok}, r_{ok}, c_{ok}, v_{ok}, o\}_{o,k}
\]

- \(F^S\): `maskmem_features`
- \(p^S\): `obj_ptr`
- \(z^S\): `object_score_logits`
- \(m^S\): `pred_masks` 또는 last-mask task-space auxiliary
- \(r\): absolute `frame_idx`와 current frame에 대한 relative age
- \(c\): conditioning/non-conditioning flag
- \(v\): slot/object validity mask
- \(o\): external canonical object ID

Translator는 raw SAM dictionary가 아니라 canonical state를 받아 다음을 출력한다.

\[
\hat a_t = T_{S\rightarrow T}(b_t; \mathcal C_T)
\]

Target contract \(\mathcal C_T\)에는 다음이 포함된다.

```text
component names and semantic roles
shape / axis order / dtype / device
object and slot capacity
conditioning vs non-conditioning storage rule
frame-index and temporal-age convention
validity/padding rule
target-generated positional constants
target injection invariants
checkpoint commit + config hash + schema version
```

**Contract correctness의 정의**는 “tensor shape가 맞아 함수가 실행된다”가 아니다. 동일 target checkpoint에서 native state를 export 후 그대로 inject했을 때, 과거 frame을 재처리하지 않고도 이후 rollout이 native continuation과 numerical-equivalent해야 한다.

### 1.2 동일 shape와 동일 semantics

SAM 2.1 Tiny↔Large는 memory tensor shape가 같을 가능성이 높다. 그러나 서로 다른 backbone/decoder가 만든 64-d memory channel과 256-d pointer coordinate가 같은 의미 basis라는 보장은 없다.

- **same shape**: assignment, concatenation, attention call이 가능하다.
- **same semantics**: target readout이 해당 tensor를 native state와 같은 방식으로 해석해 같은 후속 prediction을 만든다.

Direct Copy가 후속 J&F·identity에서 oracle과 같아야만 semantics까지 호환되었다고 판단한다. Tensor correlation이나 MSE만으로 이 결론을 내리지 않는다.

### 1.3 component별 기본 처리

| Component | 기본 처리 | 이유 | 대안/ablation |
|---|---|---|---|
| `maskmem_features` | **translate**; Direct Copy는 baseline | checkpoint별 representation basis 차이 가능 | identity, Linear/Ridge, residual MLP |
| `obj_ptr` | **translate**; feature와 별도 head | high-level object identity이며 64-d spatial feature와 통계·역할이 다름 | copy, cosine/contrastive-aligned map |
| `object_score_logits` | **calibrate/translate**하고 F/p head에도 condition | no-object semantics가 F와 p에 이미 반영됨 | scalar affine, small MLP, target no-object gating |
| `maskmem_pos_enc` | **target에서 regenerate/recompute** | frame/object 간 constant이고 target positional basis의 일부 | source copy, learned translate는 PE ablation만 |
| temporal PE / pointer tpos PE | **target에서 regenerate** | 공식 read path가 frame distance와 target parameter로 생성 | metadata corruption ablation 외 번역하지 않음 |
| frame index, cond flag, order, validity | **copy + canonicalize + validate** | control metadata이지 learned feature가 아님 | slot resampler가 필요한 pair만 target slot policy로 재배치 |
| object ID mapping | **external ID로 preserve** | translator가 새 identity를 발명하면 안 됨 | source/target internal index는 injection 때 재생성 |
| `pred_masks` | task-space에서 **copy/resize**, learned translation 금지 | 모델 공통의 관측 가능한 segmentation logit; round-trip/Last-Mask에 필요 | target memory encoder로 regenerate는 short replay에 해당 |
| target-only constants | **regenerate** | target model의 contract/parameter | 절대 source constant로 덮어쓰지 않음 |

### 1.4 frozen-model 기본 설정과 gradient

기본 학습에서는 source와 target의 모든 parameter를 freeze한다.

1. Source prefix run으로 \(b_t\)를 만들고 `detach`한다.
2. Target prefix run으로 oracle \(a_t\), oracle readout/mask trajectory를 만들고 `stop_gradient`한다.
3. Student target forward는 parameter를 freeze하되 autograd graph를 유지한다. 그래야 downstream loss의 gradient가 target memory-attention/decoder 연산을 통과해 \(\hat a_t\)와 translator로 돌아온다.
4. Optimizer가 갱신하는 parameter는 translator뿐이다.

공식 predictor의 `@torch.inference_mode()` API를 그대로 사용하면 downstream gradient가 끊긴다. A에게 inference adapter와 별도로 **differentiable frozen-target step/readout API**를 요청해야 한다.

### 1.5 방향 비대칭

**[가설] Tiny/Small→Large**는 source가 버린 appearance/detail/identity evidence를 translator가 복원할 수 없는 정보 병목이 있다. Native Large state에 대한 perfect reconstruction을 성공 조건으로 두지 않는다. 이 방향은:

- translation-only의 upper bound가 낮을 수 있다.
- 최근 frame 1–4개를 Large로 처리하는 short replay가 결손 정보를 보충할 수 있다.
- uncertainty/presence가 나쁜 switch에서 selective replay가 필요할 수 있다.

**[가설] Large→Tiny/Small**은 풍부한 state를 작은 target-compatible subspace로 압축하는 문제라 상대적으로 쉽지만, target이 수용하지 않는 source detail을 그대로 전달하면 오히려 readout shift가 생길 수 있다.

두 방향의 translator parameter는 공유하지 않는다. `T_Tiny→Large`와 `T_Large→Tiny`는 별도 model이며 inverse consistency를 강제하지 않는다. 방향 공유는 최종 compression ablation일 뿐 기본안이 아니다.

### 1.6 leakage 방지

다음은 hard invariant다.

- video split을 먼저 하고, 같은 video의 object/switch point가 서로 다른 split으로 가지 않게 한다.
- \(b_t\)와 \(a_t\)는 frame `1…t`만 처리해 만든다.
- target student는 injection 뒤 frame `≤t`의 image encoder나 memory encoder를 호출할 수 없다. 호출 counter/guard로 테스트한다.
- target-native teacher의 step `t+h` label은 학습 중 causal teacher rollout으로 만들 수 있지만 student state 입력에 teacher의 future state를 넣지 않는다.
- normalization mean/std, ridge sufficient statistics, hard-negative bank, loss-weight tuning은 train split에서만 구한다.
- validation은 architecture/loss 선택에만 사용하고 test는 최종 한 번만 사용한다.
- event-based sampling에 future GT를 사용했다면 이는 **offline sampler의 strata 정의**일 뿐 translator input feature가 아니다. test 결과는 natural/uniform slice와 event oracle slice를 따로 보고한다.
- 공식 predictor가 전체 video frames를 `inference_state["images"]`에 보관하더라도 handoff student가 과거 frame에 접근하지 못하도록 adapter 수준에서 막는다.

---

## 2. Translator architecture ladder

### 2.1 표기와 복잡도

- \(O\): object 수
- \(K_s,K_t\): source/target의 valid memory record 수
- \(H_sW_s,H_tW_t\): spatial grid token 수
- \(C_s,C_t\): spatial memory channel 수
- \(D_s,D_t\): object-pointer dimension
- \(N_F=\sum_{o,k}v_{ok}H_tW_t\): 번역할 target-grid spatial token 수
- \(N_P=\sum_{o,k}v_{ok}\): valid pointer 수
- \(h_F,h_P\): MLP hidden width

MAC은 multiply-accumulate 기준의 근사치이며 실제 latency는 kernel, dtype, transfer, serialization을 함께 측정한다.

### 2.2 비교표

| 방법 | 입력→출력 및 slot 처리 | parameter sharing | positional 처리 | Parameter 수 / MAC 근사 | 장점·inductive bias | 예상 실패 | 다음 단계 진입 조건 |
|---|---|---|---|---|---|---|---|
| **Direct Copy** | 같은 shape는 F/p/z/m을 그대로 배치; 다르면 deterministic resize/pad만. frame/object/valid metadata 보존 | parameter 없음 | target PE/tpos regenerate | `P=0`; copy 외 `0 MAC`, resize는 `O(N_F C_s)` | contract와 raw compatibility를 가장 직접 검증 | basis/scale/occlusion semantics mismatch | round-trip은 통과했지만 hard-val oracle gap이 `>1 J&F point` 또는 identity gap이 명확할 때 Linear |
| **Linear/Ridge** | 각 spatial token에 `W_F`; pointer에 `W_P`; logit에 affine. validity로 padding 제외 | 방향별 별도; F map은 object/pixel/slot 공유가 기본 | target regenerate; 필요 시 age/cond FiLM 추가 | `P=(C_s+1)C_t+C_t+(D_s+1)D_t+D_t+2`; `MAC≈N_F(C_s+1)C_t+N_P(D_s+1)D_t` | paired-state의 global affine basis alignment; closed-form/해석 가능 | nonlinear manifold, spatial/slot interaction, missing information | Stage-2 Linear이 Direct 대비 이득은 있으나 oracle gap이 남고 residual nonlinearity가 확인되면 MLP |
| **Residual MLP** | `x + W2 φ(W1 Norm(x))`; F/p/z separate heads | Linear과 동일; initially slot 공유 | target regenerate; metadata embedding은 FiLM만 | `P_F=(C_s+1)h_F+h_F+h_FC_t+C_t`, pointer도 동일식; `MAC≈N_F[(C_s+1)h_F+h_FC_t]+…` | identity 근처에서 작은 nonlinear correction | context-free라 slot 관계·spatial neighborhood를 못 봄; per-token cost 증가 | MLP가 Linear보다 `≥1 J&F` 개선하나 context/slot diagnostic상 residual이 남을 때 attention; shape mismatch면 resampler 먼저 |
| **Token resampler + projection** | spatial grid는 coordinate-preserving resize/pool 후 projection; slot은 validity/age-aware masked resampling `R∈R^{K_t×K_s}` | projection 공유; resampler는 pair/direction별 | target PE를 target grid에서 생성 | deterministic grid: `P=C_sC_t+C_t`; learned slot matrix 추가 `K_tK_s`; `MAC≈N_FC_sC_t+OH_tW_tC_tK_sK_t` | shape/slot contract mismatch를 최소 가정으로 해결 | content-dependent memory selection, object/event 정보 손실 | same-family에서는 사용하지 않음; D의 cross-architecture pair에서 `K/H/W`가 다를 때만 |
| **Component-specific translator** | F, p, z, optional mask/metadata adapter를 별도 head로 조합하는 wrapper | component 내부 공유; component 간 parameter 미공유 | PE generator와 metadata adapter는 learned head 밖 | 각 head parameter 합; MVP가 이 구조 | component별 통계와 역할을 보존; ablation 용이 | cross-component dependency를 놓침 | MVP부터 사용. presence-conditioned FiLM/slot context가 필요하면 MLP/attention wrapper로 확장 |
| **Factorized attention/Transformer** | slot summary `(GAP(F),p,z,age,cond)`에 1-layer Transformer; contextual slot output으로 F/p head를 FiLM. global pixel attention 금지 | slot Transformer는 object/slot 공유; direction별 별도 | target regenerate; age/cond를 token metadata로 입력 | base head + `(C_s+D_s+m)d + 4d²+2dd_ff+2dC_t+dD_t`; `MAC≈O[Kd(C+D)+K²d+Kdd_ff+KHW C_t]` | slot 간 dependency, cond/recent 역할, presence-context gating | source에 없는 정보는 못 만듦; overfit; latency; spatial detail interaction 부족 | MLP 이후 정량 조건을 모두 만족할 때만. 아래 §2.6 참조 |

### 2.3 Direct Copy의 정확한 두 용도

1. **Same-checkpoint round-trip copy**: native target state를 export/import한다. 모든 필드와 metadata를 보존해 injection correctness를 검증한다.
2. **Cross-checkpoint Direct Transfer**: source F/p/z/m을 target contract에 넣되 target PE/tpos/indices는 target 규칙으로 materialize한다. 이것이 학습 없는 baseline이다.

PE source-copy는 정식 Direct Transfer가 아니라 positional ablation이다. Tiny/Large PE가 tensor-equal이면 별도 실험 없이 checksum 결과만 남긴다.

### 2.4 추천 learned MVP: component-wise residual Linear/Ridge

Presence probability \(q^S=\sigma(z^S)\)를 각 component의 보조 입력으로 둔다.

\[
\hat F = F + \Delta_F([F;q^S]), \qquad
\Delta_F(x)=xW_F+b_F
\]

\[
\hat p = p + \Delta_P([p;q^S]), \qquad
\hat z = \alpha z+\beta
\]

Input/output dimension이 다른 경우 identity skip은 deterministic projection으로 바꾼다. 모든 map은 target-channel별 train-split mean/std로 standardize한 좌표에서 학습하고, output은 target scale로 되돌린다.

초기 fit은 closed-form ridge로 한다.

\[
W^*=(X^TX+\lambda I)^{-1}X^TY
\]

Spatial token 전체를 메모리에 쌓지 않고 shard를 읽으며 `XᵀX`, `XᵀY` sufficient statistics를 누적한다. \(\lambda\)는 train split만으로 fit하고 validation에서 log-grid로 고른다. 동일 family LLM KV transfer에서 linear structure와 closed-form ridge→MLP escalation이 유효했다는 최근 결과는 [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)에서 확인되지만, 이는 VOS에 대한 증거가 아니라 **architecture ladder의 참고 근거**다.

예상 SAM 2.1 contract `C=64, D=256`에서:

```text
feature head: 65×64 + 64       =   4,224
pointer head: 257×256 + 256    =  66,048
presence affine:                   2
total/direction                 =  70,274
```

Presence를 head 입력에 넣지 않는 순수 map이면 69,954 parameters다. **권장 MVP는 70,274-parameter presence-conditioned version**이며, parameter 표에는 두 수를 모두 기록한다. 이 수치는 runtime dimension 확인 후 자동 계산해야 한다.

기본 sharing은 다음과 같다.

- 같은 방향의 모든 object와 spatial position에 F map 공유
- cond/non-cond와 slot age에도 우선 공유
- F/p/z 간에는 공유하지 않음
- Tiny→Large와 Large→Tiny 간에는 공유하지 않음

Slot별 head는 표본 효율과 해석 가능성을 해치므로 residual이 age/cond별로 체계적으로 다를 때만 `cond`, `recent-1`, `older` 3-group FiLM을 추가한다.

### 2.5 다음 구조: residual MLP

Linear이 contract는 맞추지만 nonlinear residual이 남을 때만 다음을 사용한다.

\[
T_F(x)=P_0x + W_2\operatorname{GELU}(W_1\operatorname{LN}(x))
\]

- feature: `65→128→64`
- pointer: `257→256→256`
- presence: `1→16→1` 또는 scalar affine 유지
- 마지막 layer는 zero-init해 시작 시 Linear/Direct와 같게 한다.
- dropout은 처음에는 0; weight decay와 early stopping을 사용한다.

예상 parameter는 약 0.15M/direction이지만 feature MLP는 active 7-slot, 64×64 grid에서 수억 MAC이 될 수 있다. “parameter가 작다”와 “handoff latency가 작다”를 같은 뜻으로 쓰지 않는다.

### 2.6 attention/Transformer 도입 조건

도입하려면 아래 세 조건을 모두 만족해야 한다.

1. Same-checkpoint round-trip이 통과했고 metadata/PE bug가 배제되었다.
2. Stage-2 residual MLP가 Linear보다 hard-val J&F를 **최소 1 point** 개선하지만, target-native보다 **2 points 이상** 낮거나 Last-Mask→oracle gap closure가 50% 미만이다.
3. 다음 중 하나의 interaction evidence가 있다.
   - slot age/cond group별 readout residual이 전체 평균 대비 25% 이상 다르다.
   - valid slot permutation 또는 recent/old controlled drop이 1 J&F point 이상 영향을 주며 per-token MLP로 설명되지 않는다.
   - cross-architecture에서 `K_s≠K_t`이고 content-independent resampler가 replay-2보다 Pareto 열세다.

권장 attention은 모든 28K spatial token에 global attention을 거는 모델이 아니다. 각 slot에서 `GAP(F)`와 pointer, presence, age, cond flag를 합쳐 최대 수십 개의 slot token만 Transformer로 context화하고, 그 출력으로 spatial head를 FiLM한다. Spatial residual이 국소적이라고 확인될 때만 depthwise 3×3 conv를 추가한다.

**중단 조건:** attention이 MLP 대비 hard-val J&F `+0.5` 미만이거나 handoff latency가 replay-2 이상인데 정확도 우위가 없으면 폐기한다.

---

## 3. Loss 설계

### 3.1 정규화

Train split target-native component에서 channel-wise \(\mu_c^T,\sigma_c^T\)를 구하고 고정한다.

\[
S_c(x)=\frac{x_c-\mu_c^T}{\sigma_c^T+\epsilon}
\]

모든 element-wise loss는 valid object/slot/token 수와 channel 수로 나눈다. Loss 간 scale이 여전히 크게 다르면 첫 500–1,000 optimizer steps의 detached EMA로 각 loss를 나누되, validation/test 통계는 사용하지 않는다.

### 3.2 component 및 downstream losses

#### A. Normalized feature/state reconstruction

\[
\mathcal L_F = \frac{1}{|V_F|C_t}
\sum_{(o,k,h,w)\in V_F}
\operatorname{SmoothL1}\left(S(\hat F_{okhw}),S(F^T_{okhw})\right)
\]

Smooth L1을 기본으로 하고 MSE는 보고용으로 함께 기록한다. Padding/invalid slot은 제외한다. Magnitude가 downstream에 중요할 수 있으므로 per-token LayerNorm만 사용하는 loss는 기본으로 쓰지 않는다.

#### B. Object-pointer cosine + contrastive alignment

\[
\mathcal L_{cos}=\frac{1}{|V_P|}\sum_{ok\in V_P}
\left(1-\cos(\hat p_{ok},p^T_{ok})\right)
\]

\[
\mathcal L_P=\mathcal L_{cos}+0.25\mathcal L_{InfoNCE}
\]

InfoNCE positive는 같은 video/object/frame의 target pointer다. Negative는 다른 canonical object ID로 제한하고, 같은 object의 다른 switch point를 false negative로 넣지 않는다. Unique object가 충분하지 않은 batch에서는 cosine만 사용한다. 시작 temperature는 0.07이며 validation에서만 조정한다.

#### C. Presence/occlusion consistency

Target oracle probability \(q^T=\sigma(z^T/\tau_p)\)를 soft label로 사용한다.

\[
\mathcal L_O=\operatorname{BCEWithLogits}(\hat z,q^T)
\]

초기 \(\tau_p=1\)이다. Ground-truth visibility가 있는 dataset은 별도 supervised presence metric을 보고하되, target-compatible state 학습의 기본 teacher는 target-native score다.

#### D. Target memory-attention readout alignment

Target의 frame `t+1` current feature를 \(x^T_{t+1}\), frozen memory read를 \(R_T\)라 한다.

\[
\mathcal L_R=\frac{1}{CHW}
\left\|S\left(R_T(x^T_{t+1},\hat a_t)\right)-
\operatorname{sg}S\left(R_T(x^T_{t+1},a_t)\right)\right\|_2^2
\]

이는 raw state의 downstream-important subspace에 직접 gradient를 준다. Oracle branch는 stop-gradient, student branch target parameter는 freeze하되 translator input까지 gradient를 통과시킨다.

#### E. Post-switch mask distillation

Translated/native state로 target이 낸 mask logits을 \(\hat M_h,M^T_h\)라 한다.

\[
\mathcal L_M^{(h)}=
0.5\,\operatorname{BCE}\left(\sigma(\hat M_h/\tau_m),\sigma(M^T_h/\tau_m)\right)
+0.5\,\operatorname{DiceLoss}\left(\sigma(\hat M_h),\sigma(M^T_h)\right)
\]

초기 \(\tau_m=1\). Annotation이 있는 frame에서는 작은 weight의 GT BCE+Dice를 추가할 수 있지만, teacher error와 translator error를 섞지 않도록 별도 항으로 기록한다.

#### F. Multi-step rollout

Student target이 자기 예측을 새 memory로 쓰도록 하고, teacher target도 native state에서 causal rollout한다.

\[
\mathcal L_{roll}=\frac{1}{\sum_{h=1}^{H}\gamma^{h-1}}
\sum_{h=1}^{H}\gamma^{h-1}\mathcal L_M^{(h)}
\]

초기 `H=3`, `γ=0.8`; 안정화 후 `H=5`로 늘린다. 20-frame metric을 그대로 full BPTT하지 않는다. 메모리 한도를 넘으면 truncated BPTT를 사용하되 어느 frame에서 detach했는지 기록한다.

#### G. Temporal consistency

정지 mask를 강제하면 motion을 억제하므로 teacher-relative temporal difference를 사용한다.

\[
\mathcal L_{temp}=\frac{1}{H-1}\sum_{h=2}^{H}
\left\|[(\hat P_h-\hat P_{h-1})-(P^T_h-P^T_{h-1})]\odot W_h\right\|_1
\]

여기서 \(P=\sigma(M)\), \(W_h\)는 valid/visibility mask다. Occlusion transition은 별도 strata로 유지하고 무조건 smoothing하지 않는다.

#### H. Regularization

- same-dimension residual map의 identity deviation `||ΔW||²`
- standard weight decay
- output channel mean/std가 target training distribution에서 과도하게 벗어나는 moment penalty(필요할 때만)
- invalid/padded slot output이 0인지 확인하는 hard assertion

### 3.3 총 loss와 시작 weight

\[
\mathcal L_{total}=
\lambda_F\mathcal L_F+
\lambda_P\mathcal L_P+
\lambda_O\mathcal L_O+
\lambda_R\mathcal L_R+
\lambda_{roll}\mathcal L_{roll}+
\lambda_{temp}\mathcal L_{temp}+
\lambda_{reg}\mathcal L_{reg}
\]

| Stage | `λF` | `λP` | `λO` | `λR` | `λroll` | `λtemp` | `λreg` | 목적 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1. paired-state pretrain | 1.0 | 0.25 | 0.10 | 0 | 0 | 0 | `1e-4` | 안정적인 basis alignment |
| 2. one-step downstream fine-tune | 0.25 | 0.25 | 0.10 | 1.0 | 1.0 (`H=1`) | 0.05 | `1e-4` | target readout/mask 직접 최적화 |
| 3. multi-step rollout fine-tune | 0.10 | 0.20 | 0.10 | 0.5 | 1.0 (`H=3→5`) | 0.10 | `1e-4` | drift/recovery 최적화 |

이 값은 **[설계] 시작값**이다. 각 loss를 element count와 train-only scale로 정규화한 뒤 사용한다. Weight search는 한 번에 하나만 0.5×/2×로 바꾸고 validation hard-event J&F와 identity를 기준으로 고른다.

### 3.4 단계적 학습 판단

세 단계가 모두 항상 필요한 것은 아니다.

1. **Stage 1은 필수.** Linear/Ridge closed-form 또는 residual MLP initialization을 안정화하고 contract bug를 빠르게 찾는다.
2. **Stage 2는 필수.** Reconstruction만으로 연구 목표를 판단하지 않기 때문이다.
3. **Stage 3은 조건부.** one-step은 좋지만 `+5/+20` 성능이 빠르게 무너지거나 repeated-switch drift가 있을 때만 비용을 지불한다.

### 3.5 gradient 흐름

| Loss | 적용 component/path | Source | Oracle target | Student target ops | Translator |
|---|---|---|---|---|---|
| `LF` | spatial memory | detach | target state stop-grad | 사용 안 함 | update |
| `LP` | object pointer | detach | pointer stop-grad | 사용 안 함 | update |
| `LO` | presence logit 및 F/p conditioning | detach | logit stop-grad | 사용 안 함 | update |
| `LR` | target memory-attention readout | detach | full stop-grad | params frozen, input-gradient 통과 | update |
| `LM/Lroll/Ltemp` | target decoder와 future memory writes | detach | teacher trajectory stop-grad | params frozen, graph 유지 | update |

### 3.6 reconstruction과 downstream이 불일치할 때의 해석

| 관찰 | 해석 | 행동 |
|---|---|---|
| MSE 낮음, downstream 나쁨 | 중요한 readout subspace·metadata·PE가 틀리거나 small error가 rollout에서 증폭 | 먼저 round-trip/PE/slot order 재검증, 이후 `LR/LM` 강화; 곧바로 큰 translator 금지 |
| MSE 높음, downstream 좋음 | target state의 null/insensitive subspace가 다름 | 성공으로 간주; reconstruction을 더 낮추기 위해 downstream을 희생하지 않음 |
| 둘 다 나쁨 | contract 오류, capacity 부족, 정보 병목 중 하나 | same-model test→Direct→direction comparison→MLP/hybrid 순으로 분리 |
| 둘 다 좋음 | representation과 behavior가 함께 정렬 | latency/Pareto/repeated-switch 검증으로 이동 |

---

## 4. Paired-state 학습 데이터 설계

### 4.1 생성 절차

각 video/object/switch `t`에 대해 다음을 수행한다.

1. Dataset의 canonical video ID와 object IDs를 고정하고 source/target에 **동일한 prompt protocol**을 준다.
2. Frozen source를 frame `1…t`까지 causal run해 `SourceState b_t`를 export한다.
3. Frozen target을 같은 prefix와 prompt로 run해 `TargetState a_t`를 export한다.
4. 두 state를 external object ID와 absolute frame index로 join한다. Internal object index로 join하지 않는다.
5. A의 target selection policy로 continuation-closed record set을 계산한다. Cond frames, 앞으로 다시 참조될 수 있는 recent history, frame direction을 보존한다.
6. `StateSpec`, checkpoint SHA/config hash, dataset split, video/object/switch IDs, event tags를 함께 저장한다.
7. `maskmem_pos_enc`와 target temporal PE처럼 regenerate 가능한 constant는 sample마다 저장하지 않는다.
8. Source/target state 모두에 schema validator를 실행하고 invalid/padding mask를 저장한다.
9. 작은 audit subset에는 FP32 reference도 저장해 BF16/FP16 quantization 영향과 serialization round-trip을 측정한다.
10. Dataset 생성과 translator fit의 manifest를 immutable하게 남긴다. 다른 checkpoint/config의 pair를 같은 shard에 섞지 않는다.

### 4.2 switch point sampling

Training switch sampler의 시작 mixture를 다음처럼 둔다.

- **50% uniform/length-stratified:** valid `t`를 prefix length quartile별로 균등 추출한다.
- **35% hard event:** occlusion 직전/중간, reappearance 직전/직후, distractor 접근, fast motion, 큰 scale/appearance change 주변을 oversample한다.
- **15% memory-boundary:** valid slot 수가 1…K로 달라지는 warm-up, slot eviction/stride boundary, long-prefix switch를 추출한다.

각 video/object에서 지나치게 많은 인접 `t`를 뽑지 않는다. 한 epoch/shard에서 video당 switch 수를 cap하고 object-balanced sampler를 사용한다. Validation/test에는 oversampling weight를 적용하지 않고 다음 두 결과를 모두 낸다.

- natural/uniform switch distribution
- predefined hard-event slice

Hard event는 가능하면 GT mask trajectory로 재현 가능하게 정의한다.

- disappearance/occlusion: visible area가 0 또는 threshold 아래로 감소
- reappearance: 연속 absent 구간 뒤 visible area 회복
- fast motion: mask centroid displacement / object scale가 상위 quantile
- scale change: area ratio 변화 상위 quantile
- distractor: 다중 object 근접/overlap 또는 evaluator가 제공하는 distractor tag

Quantile과 threshold는 train split에서 고정하고 test에 다시 맞추지 않는다. Future GT로 event를 찾는 것은 offline sampling/evaluation strata 정의에만 허용하며 translator input에 event oracle을 넣지 않는다.

### 4.3 split 원칙

- Dataset 공식 train/val/test split이 있으면 그대로 사용한다.
- 공식 train만 제공되는 추가 corpus는 **original video/group 기준**으로 train/internal-val을 나눈다. Derived clips가 같은 원본 video에서 왔다면 같은 group이다.
- 같은 video의 서로 다른 object, switch point, augmentation은 반드시 같은 split에 둔다.
- 동일 scene의 near-duplicate video를 찾을 수 있으면 hash/metadata로 group화한다.
- Test video에서 normalization statistics, ridge λ, architecture, replay-k, loss weight를 선택하지 않는다.
- Cross-dataset generalization은 `train on A → test on B`로 별도 표를 만들고 in-domain과 섞지 않는다.

### 4.4 slot 수·validity가 다른 사례

Canonical storage는 ragged record를 허용하되 batch 시 padding한다.

```text
features        [B, Omax, Kmax, C, H, W]
obj_ptr         [B, Omax, Kmax, D]
presence_logits [B, Omax, Kmax, 1]
frame_idx       [B, Omax, Kmax]
is_conditioning [B, Omax, Kmax]
valid           [B, Omax, Kmax]
object_valid    [B, Omax]
```

- Padding은 zero로 하고 모든 loss/attention에서 `valid=False`를 hard-mask한다.
- Source/target의 slot 수가 같아도 실제 valid 수는 early prefix·missing frame·stride 때문에 다를 수 있다.
- Pair alignment는 array index가 아니라 `(object_id, frame_idx, conditioning role)`로 한다.
- Target-native record가 source에 없으면 translator가 임의 복제하지 않는다. Same-family에서 이 상황은 prompt/state-policy mismatch일 수 있으므로 먼저 data error로 분류한다.
- Cross-architecture의 `K_s≠K_t`는 D가 제공하는 canonical memory-role mapping과 §2의 resampler가 처리한다.

### 4.5 object 수가 다른 사례

- Prompt로 등록된 external object registry를 truth로 둔다.
- Object가 frame에서 보이지 않아도 object slot 자체는 제거하지 않고 `presence/validity`로 표현한다.
- `t` 이전에 아직 prompt로 도입되지 않은 object는 source/target 양쪽에서 `object_valid=False`다. Translator가 새 object를 만들지 않는다.
- Source/target 중 한쪽만 object를 갖는다면 prompt protocol 또는 predictor adapter mismatch다. 학습 pair로 쓰지 말고 quarantine report를 남긴다.
- 다중 object에서 predictor가 per-object inference를 하더라도 object ID mapping과 overlap constraint metadata가 injection 후 보존되는지 round-trip으로 검증한다.

### 4.6 precision과 저장 용량

권장 기본 저장:

- `maskmem_features`: BF16 또는 FP16
- `obj_ptr`: FP16/BF16; cosine audit subset은 FP32
- `object_score_logits`: FP16
- frame/object/flags: integer/bool
- `maskmem_pos_enc`: 저장하지 않고 target에서 regenerate
- `pred_masks`: round-trip/Last-Mask용 last frame 또는 필요한 cond records만 저장; 모든 slot 저장은 기본값 아님

Active record만 저장한다고 가정한 component pair의 대략적 byte 수는:

\[
B_{pair}\approx 2O\{K(HWC\,b_F+D\,b_P+b_z)+B_{meta}\}+B_{mask}
\]

예상 SAM 2.1 `K=7,H=W=C=64,b_F=2 bytes`이면 spatial feature만:

- 한 model/object: 약 3.5 MiB
- source+target/object/pair: 약 7.0 MiB
- 모든 7개 `256×256` FP16 mask까지 넣으면 pair당 약 1.75 MiB 추가

따라서 100K object-switch pair는 raw feature만으로도 약 700 GiB 규모다. 실제 수치는 A의 continuation-closed record 수와 object packing을 넣어 manifest 생성 시 계산한다.

저장 정책:

- 1–2 GiB 정도의 immutable shard와 index manifest 사용
- constant PE 중복 금지
- channel statistics와 ridge sufficient statistics는 streaming 계산
- switch가 인접한 pair는 같은 prefix state history를 중복 저장하지 않는 indexed record store 검토
- INT8/PQ는 audit subset에서 post-switch J&F `≤0.1 point` 차이를 확인한 뒤에만 사용
- data loader가 dtype을 FP32로 올리더라도 disk byte와 wire-transfer byte를 따로 기록

### 4.7 class/video imbalance

- sampling unit은 spatial token이 아니라 먼저 video→object→switch다. 큰 object와 긴 video가 token 수로 dataset을 지배하지 않게 한다.
- loss는 object/switch별 평균을 낸 뒤 batch 평균한다.
- hard-event oversampling은 sampler weight로만 하고 결과 metric을 재가중하지 않는다.
- rare event는 per-event minimum quota를 두되 동일 video의 반복 switch를 독립 video처럼 세지 않는다.
- confidence interval은 video-cluster bootstrap으로 계산한다.

### 4.8 offline state-pair vs online rollout

| 방식 | 장점 | 단점 | 권장 용도 |
|---|---|---|---|
| Offline paired state | deterministic, source/target forward 비용 재사용, Ridge/Linear에 최적, debugging 쉬움 | storage 큼, self-generated shifted state를 못 봄, multi-step gradient 없음 | Stage 1 전체와 Stage 2 일부 |
| Online one-step | current target readout/mask loss 가능, storage 감소 | image/target forward 비용, API 복잡, reproducibility 관리 필요 | Stage 2 hard-event subset |
| Online rollout | exposure bias와 drift를 직접 최적화 | 가장 비싸고 memory-intensive, inference-mode API를 수정해야 함 | one-step 성공 후 Stage 3 subset |

권장 조합은 offline Ridge/MLP pretrain → online one-step fine-tune → drift가 확인될 때만 short rollout이다.

---

## 5. State-component ablation

### 5.1 최소 core matrix

모든 조합을 factorial하게 돌리지 않는다. 아래 nested 비교는 **방향별**로 수행한다. B가 제공하는 Reset/Last-Mask/Replay/Oracle은 중복 실행하지 않는다.

| ID | Translator/state | Loss | Replay | 식별하는 질문 |
|---|---|---|---:|---|
| C0 | Cross-checkpoint Direct Copy, target PE | 없음 | 0 | raw semantics 호환성 |
| C1 | Linear **F only**; pointer token gate off, z는 기록만 | `LF` | 0 | feature만 충분한가 |
| C2 | Linear F + translated `obj_ptr` | `LF+LP` | 0 | pointer의 identity 기여: C2−C1 |
| C3 | Full Linear F+p+presence-conditioned heads | Stage 1 | 0 | presence/complete component 기여: C3−C2 |
| C4 | Full Linear | Stage 2 one-step | 0 | downstream loss의 실제 기여: C4−C3 |
| C5 | Full Linear | Stage 3 rollout | 0 | multi-step loss가 +5/+20/drift를 만드는가: C5−C4 |
| C6 | C4, cond + **recent 3** valid records만 | Stage 2 | 0 | recent memory의 가치 |
| C7 | C4, cond + record 수를 맞춘 **older 3** | Stage 2 | 0 | old memory/long-term evidence의 가치: C7 vs C6 |
| C8 | best translation-only | best loss | 1 | hybrid의 초기 crossover |
| C9 | best translation-only | best loss | 2 | hybrid Pareto |
| C10 | best translation-only | best loss | 4 | hybrid Pareto |

`F only`에서 `obj_ptr`를 zero tensor로 속이는 대신 A의 read adapter가 pointer token을 attention에서 제외하는 component gate를 제공하는 것이 바람직하다. 그런 gate가 불가능하면 target `no_obj_ptr`/zero intervention을 둘 다 기록해 결과가 intervention 방식에 의존하는지 확인한다.

### 5.2 positional encoding 최소 실험

먼저 Tiny/Small/Large의 runtime `maskmem_pos_enc` checksum과 max-absolute difference를 비교한다.

- tensor-equal 또는 dtype tolerance 내 같음: target-regenerate를 고정하고 별도 training run을 하지 않는다.
- 다름: C4를 기준으로 `(a) source copy`, `(b) target regenerate`, `(c) affine-translated PE` 세 조건을 비교한다. Default `(b)`가 이미 C4이므로 추가 run은 2개다.
- temporal PE는 source tensor가 존재하지 않으므로 translate 대상이 아니다. Frame-age corruption/permutation으로 metadata 중요도만 검사한다.

### 5.3 sharing ablation은 조건부

기본은 slot 공유, component/direction 비공유다.

- C4 residual을 `cond`, `recent-1`, `older`로 나눴을 때 normalized readout error가 전체 평균보다 25% 이상 차이나면 3-group FiLM head 1개를 추가한다.
- Two-direction 개별 model이 완성된 뒤에만 direction token을 가진 joint translator 1개를 학습한다. 두 개별 model 대비 parameter는 줄지만 어느 방향이 `>0.5 J&F point` 나빠지면 공유 불가로 결론낸다.
- Component head 공유는 F와 p의 차원/역할이 달라 기본적으로 하지 않는다. 공통 latent를 쓰는 실험은 cross-architecture scaling 단계로 미룬다.

### 5.4 MLP와 attention의 위치

- C4 Linear이 C0 Direct보다 hard-val `+0.5 J&F` 미만이면 먼저 contract·data alignment를 조사한다. 곧바로 MLP를 추가하지 않는다.
- Linear이 의미 있는 이득을 보이나 oracle gap이 남으면 동일 C4 설정의 residual MLP 1개를 추가한다.
- Attention은 §2.6 조건을 만족할 때만 MLP와 동일 Stage-2 조건으로 1개를 추가한다.

### 5.5 replay 확장 중단 규칙

- Translation+replay-1/2/4 중 어느 것도 translation-only보다 J&F `+1` 이상 또는 recovery/identity 개선을 만들지 못하면 hybrid를 중단한다.
- replay-4가 여전히 개선 중이고 B의 replay-8/16 frontier보다 latency 이점이 남을 때만 hybrid-8을 실행한다.
- Hybrid가 같은 k의 replay-only에 정확도 우위가 없고 translator latency/bytes만 추가하면 hybrid는 Pareto-dominated로 폐기한다.

---

## 6. 평가 및 판정 기준

### 6.1 B baseline과 결합할 공통 protocol

동일 `(dataset, video, object, switch t, direction, prompt protocol, dtype, device)` key로 다음 결과를 join한다.

```text
Target Reset
Last-Mask
Replay-1/2/4/8/16
Full Replay / Target-Native Oracle
Direct Transfer
Learned Translator
Translation + Short Replay
```

각 method는 pre-switch source trajectory를 공유하고 handoff 방식만 다르게 한다. Full Replay와 target-native oracle이 numerical-equivalent인지 B/A가 먼저 확인한다.

### 6.2 정확도·연속성 metric 정의

- **J&F, J, F:** 전체와 event slice를 분리한다.
- **`J&F@+1/+5/+20`:** 정확히 frame `t+1,t+5,t+20`의 score. Video가 짧아 해당 frame이 없으면 missing으로 두고 분모를 보고한다.
- **`AUC-20`:** frame `t+1…t+20`의 평균 J&F. 즉시 shock와 초기 recovery를 안정적으로 요약한다.
- **Handoff gap:** `G_h = J&F_oracle,h − J&F_method,h`.
- **Switching shock:** `Shock_5 = mean_{h=1..5} G_h`. 단순 pre/post score drop은 frame 난이도와 model capacity 차이를 섞으므로 보조치로만 둔다.
- **Recovery length:** gap이 `≤1 J&F point`가 된 뒤 3 frame 연속 유지되는 최초 `h`. Horizon 안에 회복하지 못하면 right-censored로 보고하고 median/회복률을 함께 낸다.
- **Identity break:** multi-object frame마다 mask-IoU Hungarian assignment를 만들고 predicted object ID가 다른 GT identity로 바뀌는 event를 센다. `events/100 switches`, affected-video 비율, reappearance 직후 break를 분리한다.
- **Occlusion-reappearance:** disappearance 이전 state를 갖고 absent interval 뒤 첫 1/5/20 visible frames의 J&F, recovery length, distractor error를 보고한다.
- **Repeated-switch drift:** 동일 sequence에서 alternating switch count `n`별 oracle gap, state/readout error, J&F slope(points/switch)를 보고한다. 각 segment의 native target prefix state가 reference다.

Recovery threshold `1 point/3 frames`는 **[설계] 시작 정의**다. B의 native-run noise/quantization이 1 point에 근접하면 validation에서 threshold를 한 번 조정하고 고정한다.

### 6.3 효율 metric

Handoff critical path는 다음을 합친다.

```text
source state export/serialization
device or network transfer
translator forward
target materialization/injection
optional replay
target first post-switch frame ready
```

각 항목과 end-to-end를 모두 측정한다.

- synchronized median, p90, p95 latency; warm-up 제외 규칙 고정
- translator MACs/FLOPs와 실제 kernel time
- peak allocated/reserved VRAM
- source state raw bytes, serialized/wire bytes, target materialized bytes
- translator checkpoint parameter bytes
- optional energy/cost는 같은 hardware condition에서만

Translation의 계산이 작아도 7-slot spatial state 전송이 병목일 수 있으므로 `compute-only latency`를 end-to-end latency로 오인하지 않는다.

### 6.4 통계

- video를 cluster로 한 paired bootstrap 95% CI
- direction별 별도 CI
- natural/uniform과 hard-event 결과 별도
- switch point 수가 많아도 독립 video 수를 표본 수처럼 부풀리지 않음
- method 비교는 같은 switch key의 paired difference로 계산
- architecture/loss 선택은 validation, test는 frozen config

### 6.5 정량 Go/No-Go gate

아래 threshold는 **[설계] 사전등록 시작값**이며 pilot 분산을 본 뒤 test 전에 한 번만 고정한다. J&F는 0–100 points 기준이다.

#### Gate 0 — contract

동일 checkpoint export→inject가:

- 20-frame continuation에서 J&F 차이 `≤0.1 point`
- binary mask 동일률 100% 또는 logit tolerance 내 allclose
- object/slot/frame metadata exact match
- target의 과거 frame (`≤t`) encoder 호출 0회

를 만족하지 못하면 **No-Go: translator 학습 중단, A interface 수정**이다.

#### Gate 1 — learned same-family translator 필요성

- Direct Copy가 oracle과 hard-val `≤1 J&F point`, identity break 차이 `≤5% relative`이면 **same-family learned translator는 불필요**하다. Direct handoff를 결과로 남기고 cross-architecture로 이동한다.
- Direct Copy gap이 그보다 크면 Linear/Ridge로 진입한다.

#### Gate 2 — Linear/MLP 가치

Learned translator가 hard-event validation에서:

- Last-Mask 대비 `≥2 J&F points`
- video-cluster bootstrap CI lower bound `>0`
- Last-Mask→oracle gap의 `≥50%` closure
- identity break `≥20% relative` 감소

중 **세 가지 이상**을 만족하고 accuracy–latency Pareto에서 non-dominated면 Go다. 평균 J&F 하나만 만족하면 보류하고 event/identity 원인을 분석한다.

#### Gate 3 — attention

§2.6의 interaction evidence와 threshold를 모두 만족해야 한다. Attention이 MLP보다 `+0.5 J&F` 미만이거나 replay-2보다 느리면서 정확도 우위가 없으면 No-Go다.

#### Gate 4 — hybrid

Translation+replay-k가:

- replay-k only보다 hard-event J&F `≥1 point` 높거나,
- 같은 J&F(±1 point)에서 handoff latency가 `≥2×` 낮거나,
- translation-only 대비 recovery/identity를 유의하게 개선

하는 k가 하나라도 있으면 hybrid를 유지한다. 그렇지 않으면 translation-only 또는 replay-only가 답이다.

#### 연구 아이디어의 실패 조건

다음이 동시에 성립하면 해당 pair/scenario에서 memory translation 가치를 주장하지 않는다.

- best learned translator가 Last-Mask보다 `<1 J&F point` 개선이고 CI가 0을 포함
- translation/hybrid 전부 replay-1/2/4 frontier에 Pareto-dominated
- hard occlusion/reappearance에서도 identity/recovery 이득 없음

Tiny→Large만 실패하고 Large→Tiny가 성공하면 방향 비대칭 결과로 보고한다. 양방향 same-family가 실패해도 complete state contract가 맞다면 cross-architecture 한 pair로 문제 설정을 재검증할 수 있으나, cross-architecture도 동일하게 실패하면 core claim은 No-Go다.

---

## 7. 실행 가능한 실험 순서

| 단계 | 검증 가설 | 필요한 입력 | 구현/비교 | 핵심 metric | 예상 산출물 | 성공/실패 다음 행동 |
|---|---|---|---|---|---|---|
| **E0. Schema census + same-checkpoint round-trip** | export state가 continuation-closed이고 injection이 native를 보존 | A의 state dump/API, Tiny/Large checkpoint+commit, 3개 이상 smoke videos | Translator 없음; native vs export→inject | logit diff, +1/+5/+20 J&F, past-frame call count | schema manifest, closure test, failing-field trace | 통과→E1; 실패→A와 contract 수정, 학습 금지 |
| **E1. Cross-checkpoint Direct Copy** | same shape가 downstream semantics까지 호환되는가 | E0 API, B evaluator | Tiny→Large/Large→Tiny Direct, Reset, Last-Mask, Oracle | hard J&F, Shock-5, identity, latency/bytes | direct compatibility table | oracle와 ≤1 point→learned same-family 생략; 아니면 E2 |
| **E2. Paired-state mini-shard + Ridge** | global affine alignment가 존재하는가 | train-only 100–수천 pair의 schema-valid shard | component-wise Ridge, Direct | normalized recon, pointer cosine, one-step readout | channel stats, λ sweep, parameter/latency report | readout 개선→E3; 무개선→alignment/PE/metadata audit 후 MLP 판단 |
| **E3. Full Linear + component ablation** | F/p/presence 중 실제 continuity component는 무엇인가 | E2 full shards, A component gates | C1/C2/C3, C0 | +1/+5, identity, occlusion recovery | component contribution table | full 우위→E4; feature-only 동일→필수 state 주장 축소 |
| **E4. One-step downstream fine-tune** | readout/mask loss가 reconstruction-only보다 behavior를 개선 | differentiable frozen-target API, frame `t+1` | C3 vs C4 | `LR`, mask distill, J&F@+1/+5 | loss ablation, readout-error plot | +0.5 이상/identity 개선→E5; 무개선→recon 충분 또는 teacher-path 문제 조사 |
| **E5. Residual MLP gate** | remaining map이 nonlinear지만 context-free한가 | E4 dataset/API | Linear C4 vs residual MLP C4 | hard J&F, gap closure, latency | linear-vs-MLP Pareto | MLP +1 이상→best로 채택; <0.5→Linear 유지 |
| **E6. Multi-step/slot/drift** | one-step 성공이 rollout에서 유지되는가 | online causal rollout API | best Stage2 vs C5, C6/C7, repeated switch | +5/+20, recovery, drift slope | slot-age/loss-stage plots | drift→Stage3 유지; stable→one-step으로 비용 절약 |
| **E7. Hybrid frontier** | short replay가 source 정보 병목을 싸게 보충하는가 | B replay evaluator + translated injection | C8/C9/C10 vs replay-only 1/2/4 | J&F-latency Pareto, identity, bytes | direction별 frontier | hybrid 우위→정식 방법; 열세→폐기 |
| **E8. Attention gate** | slot/context interaction이 MLP의 남은 오차 원인인가 | §2.6 evidence | slot-summary Transformer 1개 vs best MLP | hard J&F, readout residual, latency | attention decision memo | +0.5 미만/지연 큼→폐기; 유의 개선→ablation 1회 |
| **E9. D handoff** | canonical interface가 architecture-independent한가 | frozen StateSpec/API와 best component heads | D의 source/target adapters + resampler | schema coverage, round-trip, initial Direct | cross-architecture integration spec | D state contract 확인 후 pair-specific training |

`E2`의 pair 수는 hardware/storage가 확인되지 않아 고정하지 않는다. 먼저 mini-shard로 learning curve를 만들고, validation loss가 pair 수에 따라 계속 감소할 때만 확대한다.

---

## 8. 팀 간 interface

### 8.1 A에게 받아야 할 것

1. **Pinned state schema manifest**
   - repo commit, checkpoint SHA, config hash
   - field name, semantic role, shape, axis, dtype, device, optionality
   - Tiny/Small/Large 실제 runtime dump
2. **State extraction API**
   - external object ID와 absolute frame index 유지
   - continuation-closed cond/non-cond history export
   - no hidden past-frame reference
3. **State injection/materialization API**
   - target PE/tpos regeneration
   - target internal object mapping 재생성
   - validity/slot order validation
4. **Round-trip harness**
   - same-checkpoint numerical continuation comparison
   - encoder call audit
5. **Differentiable frozen-target API**
   - `read_memory()`와 `step()`에서 target parameter freeze, state input gradient 유지
6. **Component gate/hook**
   - pointer tokens, individual spatial slots, presence gating을 target weight 변경 없이 ablate

A가 확인해야 할 특히 중요한 항목은 `pred_masks`와 original prompt dictionary가 pure continuation에 필요한지, cond frame을 몇 개 보존해야 continuation-closed인지, stored `object_score_logits`가 injection 뒤 어떤 경로에서 다시 읽히는지다.

### 8.2 B에게 받아야 할 것

- 공통 `EvaluationCase` key와 deterministic prompt/switch protocol
- Reset, Last-Mask, Replay-1/2/4/8/16, Full Replay/Oracle 결과
- per-frame/per-object J, F, J&F raw table
- hard-event tags 또는 tag 생성 API
- identity break/Hungarian assignment evaluator
- latency timer, GPU sync, FLOPs, peak VRAM, transfer-byte hooks
- video-cluster bootstrap과 Pareto plotting input schema

C는 B의 baseline을 재정의하지 않고 `Direct/Learned/Hybrid`가 같은 evaluator를 호출하게 한다.

### 8.3 D에게 넘길 architecture-independent interface

- raw SAM field 이름을 숨긴 canonical `StateSpec`/`CanonicalState`
- component role enum: `SPATIAL_MEMORY`, `OBJECT_TOKEN`, `PRESENCE`, `TASK_MASK`, `POSITIONAL`, `CONTROL_METADATA`
- ragged object/slot validity와 frame-age convention
- shape adapter/token resampler hook
- target-side `PositionProvider`와 `StateMaterializer`
- translator가 지원/미지원하는 component pair를 명시하는 capability manifest

D가 XMem/Cutie의 내부 memory를 이 canonical role에 매핑하고, C는 translator core가 `C/H/W/D/K`를 hard-code하지 않게 유지한다. Cross-architecture에서 source와 target component semantics가 대응하지 않으면 D가 무리한 1:1 mapping을 만들지 말고 `unmatched/target-only`로 표시해야 한다.

### 8.4 권장 dataclass/pseudocode

```python
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class StateSpec:
    model_id: str
    checkpoint_sha: str
    code_commit: str
    config_hash: str
    schema_version: str
    spatial_channels: int
    spatial_hw: tuple[int, int]
    pointer_dim: int
    max_memory_records: int
    storage_dtype: torch.dtype
    positional_policy: str      # e.g. "target_regenerate"
    direction_policy: str       # e.g. "forward"

@dataclass
class SourceState:
    spec: StateSpec
    switch_frame: Tensor        # [B]
    object_ids: Sequence[Sequence[str]]
    features: Tensor            # [B, O, K, Cs, Hs, Ws]
    obj_ptr: Tensor             # [B, O, K, Ds]
    presence_logits: Tensor     # [B, O, K, 1]
    pred_mask_logits: Optional[Tensor]
    frame_idx: Tensor           # [B, O, K]
    is_conditioning: Tensor     # bool [B, O, K]
    valid: Tensor               # bool [B, O, K]
    object_valid: Tensor        # bool [B, O]
    extra_control: Mapping[str, Tensor]

@dataclass
class TargetState:
    spec: StateSpec
    features: Tensor
    obj_ptr: Tensor
    presence_logits: Tensor
    pred_mask_logits: Optional[Tensor]
    frame_idx: Tensor
    is_conditioning: Tensor
    valid: Tensor
    object_valid: Tensor
    # PE is materialized from target model + metadata, not learned state.
    materialization_metadata: Mapping[str, Tensor]

class Translator(nn.Module):
    def forward(
        self,
        source: SourceState,
        target_spec: StateSpec,
        *,
        target_constants: Mapping[str, Tensor],
    ) -> TargetState:
        """Must not access video frames or target-native prefix state."""
        raise NotImplementedError

@dataclass
class LossBatch:
    source: SourceState
    oracle: TargetState
    future_frames: Optional[Tensor]       # train split only
    teacher_mask_logits: Optional[Tensor]
    gt_masks: Optional[Tensor]
    event_tags: Mapping[str, Tensor]

@dataclass
class LossOutput:
    total: Tensor
    terms: Mapping[str, Tensor]
    counts: Mapping[str, int]

def translator_loss(
    translated: TargetState,
    batch: LossBatch,
    frozen_target_adapter,
    weights: Mapping[str, float],
) -> LossOutput:
    ...
```

API invariant:

```text
Translator.forward()는 frame tensor, target-native a_t, future label에 접근하지 않는다.
StateMaterializer만 target PE/indices/internal dictionaries를 생성한다.
Loss 함수만 frozen target의 readout/rollout을 호출한다.
```

### 8.5 C가 직접 책임질 산출물

**코드**

- canonical state validator와 translator-side schema checks
- `DirectCopyTranslator`
- component-wise Ridge/Linear 및 sufficient-statistics fitter
- residual MLP
- conditional token resampler
- conditional slot-summary attention translator
- normalized component/downstream/rollout losses
- paired-state dataset loader, sampler, train-only normalization
- trainer, checkpoint manifest, reproducibility config
- ablation runner와 result joiner(B output schema 소비)

**실험/분석**

- 양방향 Direct/Linear/MLP table
- loss-stage ablation
- F/p/presence/position/slot ablation
- reconstruction↔readout↔J&F correlation/scatter
- J&F–latency Pareto 및 bytes–accuracy plot
- switch-relative performance curve `+1…+20`
- identity/recovery/repeated-switch drift plot
- parameter/MAC/latency/VRAM table
- failure cases와 Go/No-Go decision memo

### 8.6 C 역할을 2명이 맡을 때의 권장 분담

두 사람을 `Tiny→Large`와 `Large→Tiny`로 나누지 않는다. 방향별 분담은 같은 state contract, translator, loss, evaluator를 두 번 구현하게 만들고 방향 차이가 코드 차이와 섞인다. 두 사람 모두 양방향을 다루되 **representation 경계**를 기준으로 나눈다.

#### C1 — Translator Core & Offline Alignment

**책임 범위:** `SourceState → TargetState` 변환이 정확하고 작고 재현 가능하게 동작하도록 만드는 것.

- A의 schema manifest를 받아 `StateSpec`, `SourceState`, `TargetState`, validator를 확정한다.
- A의 raw extraction/injection API 위에 translator-side adapter와 continuation-closed state 검증을 붙인다.
- same-checkpoint round-trip 및 cross-checkpoint `DirectCopyTranslator`를 구현한다.
- paired-state loader, train-only channel statistics, shard index, Ridge sufficient-statistics fitter를 구현한다.
- component-wise Linear/Ridge와 residual MLP를 구현한다.
- 필요 조건이 충족될 때만 token resampler와 slot-summary attention을 구현한다.
- F/p/presence/slot component gate에 필요한 translator-side hook을 제공한다.
- parameter 수, MACs, translator-only latency, translator checkpoint/serialized-state bytes를 산출한다.
- D에게 넘길 architecture-independent `StateSpec`/capability manifest의 owner가 된다.

**C1의 완료 조건:** 두 방향 모두 같은 config/API로 실행되고, unit test·schema hash·parameter/MAC·translator-only latency가 포함된 versioned translator release를 C2가 그대로 불러올 수 있다.

#### C2 — Downstream Training, Ablation & Systems Evaluation

**책임 범위:** translated state가 실제 target behavior와 accuracy–latency Pareto를 개선하는지 검증하는 것.

- A의 differentiable frozen-target `read_memory()/step()` API를 소비하고 gradient-flow test를 작성한다.
- normalized reconstruction, pointer, presence, readout, mask distillation, rollout, temporal loss를 구현한다.
- Stage 1/2/3 trainer, reproducibility config, checkpoint selection rule를 구현한다.
- B의 공통 evaluator와 baseline raw result를 연결한다. B의 baseline이나 metric 자체를 재구현하지 않는다.
- C0–C10 nested ablation과 conditional PE/sharing/MLP/attention 실험을 실행한다.
- translation-only와 translation+replay-1/2/4를 B의 replay-only frontier와 비교한다.
- `J&F@+1/+5/+20`, Shock-5, recovery, identity, repeated-switch drift를 분석한다.
- end-to-end handoff latency, peak VRAM, transfer bytes와 accuracy–latency Pareto를 산출한다.
- Go/No-Go decision memo, failure case, 표·그래프의 owner가 된다.
- D의 cross-architecture adapter가 들어오면 behavior acceptance test를 수행한다.

**C2의 완료 조건:** frozen translator release 하나로 전체 experiment manifest를 재실행할 수 있고, per-case raw result·clustered CI·Pareto plot·Go/No-Go 근거가 남는다.

#### 작업별 단일 owner

| 작업 | Owner | Reviewer/의존 | 산출물 |
|---|---|---|---|
| A schema 수락·canonical state | C1 | C2가 loss/evaluator 요구사항 review | `state_spec.json`, validator tests |
| Same-checkpoint round-trip | C1 | C2가 +1/+5/+20 behavior 검증 | Gate-0 report |
| Direct/Ridge/Linear/MLP | C1 | C2가 training usability review | versioned translator checkpoints |
| Paired-state offline pipeline | C1 | C2가 split/leakage review | shard manifest, stats, Ridge fit |
| Loss와 differentiable target path | C2 | C1이 state gradient/shape review | loss unit tests, gradient audit |
| C0–C10 ablation 실행 | C2 | C1이 component intervention 검증 | ablation raw table |
| Resampler/attention 구현 | C1 | C2의 entry evidence가 선행 | conditional model release |
| Hybrid/repeated-switch | C2 | B evaluator 및 C1 translator | Pareto/drift report |
| D handoff | C1: interface / C2: behavior acceptance | D adapter | capability + acceptance report |
| 최종 method 표 | C2 draft | C1 complexity/fact check | paper-ready tables/figures |

#### 두 사람의 공통 승인 지점

1. **Gate 0:** round-trip가 통과하기 전에는 누구도 translator 학습 결과를 정식 결과로 올리지 않는다.
2. **MVP freeze:** C1이 Linear/Ridge release를 고정하고 C2가 C1–C4를 끝낼 때까지 architecture를 임의 변경하지 않는다.
3. **Escalation review:** MLP, rollout, hybrid, attention 진입은 C2의 metric evidence와 C1의 residual/complexity evidence를 함께 기록한다.
4. **Test freeze:** 두 사람이 validation에서 최종 config를 공동 서명한 뒤 test를 한 번 실행한다.

#### 첫 1주 병렬 일정

| 시점 | C1 | C2 | 공동 산출물 |
|---|---|---|---|
| Day 1 | A schema를 받아 dataclass/validator 작성 | B result schema·metric contract 확인, 실험 manifest 초안 | interface checklist |
| Day 2 | same-checkpoint round-trip + Direct Copy | synthetic state로 loss API와 gradient test 작성 | Gate-0/Direct smoke report |
| Day 3 | mini paired-state shard + train-only stats + Ridge | Stage-1/2 trainer와 evaluator join 구현 | 첫 Linear checkpoint와 재현 config |
| Day 4 | Linear 양방향 fit, component gate 검증 | C0–C4 양방향 실행 | component/loss nested table |
| Day 5 | residual·complexity 분석, MLP 필요성 판단 | hard-event/identity/Pareto 분석 | 다음 주 Go/No-Go memo |

Blocker가 생겼을 때 C1은 A의 raw SAM 내부 코드를 대신 소유하지 않고 필요한 contract violation을 재현해 A에게 넘긴다. C2도 B의 baseline evaluator를 포크하지 않고 missing metric/interface를 요청한다.

---

## 9. 최종 권고안

### 9.1 한눈에 보는 결정

| 항목 | 권고 |
|---|---|
| 가장 먼저 구현 | same-checkpoint state round-trip용 Direct Copy와 target-side PE materializer |
| 첫 learned translator | direction-specific component-wise residual Linear/Ridge; F/p separate, presence-conditioned, target PE regenerate |
| 다음 translator | residual 2-layer MLP (`F 65→128→64`, `p 257→256→256`) |
| 기본 loss | Stage 1 `LF + 0.25LP + 0.1LO`; Stage 2 `0.25LF + 0.25LP + 0.1LO + LR + LM + 0.05Ltemp` |
| 필수 state | translated F+p, presence-aware conditioning, external object ID, cond/non-cond frame history, frame age/order/validity; target-generated PE |
| 최소 ablation | C0–C10 nested matrix; PE checksum; MLP 1회는 Linear gate 통과 시 |
| attention 도입 | round-trip 통과 + MLP가 Linear보다 ≥1 향상 + oracle gap/interaction evidence가 모두 존재할 때 |
| short replay 도입 | Tiny/Small→Large 정보 병목 또는 +5/+20 drift가 있고 hybrid-1/2/4가 Pareto 개선할 때 |
| sharing | slot 공유가 기본, component/direction 비공유; residual stratification이 있을 때만 FiLM/group head |
| 핵심 판정 | reconstruction이 아니라 hard-event downstream continuity와 accuracy–latency Pareto |

### 9.2 가장 큰 기술적 위험 5개와 조기 탐지

| 위험 | 조기 탐지 | 대응 |
|---|---|---|
| **Incomplete state contract** | same-checkpoint round-trip가 +1부터 다름, 과거 encoder call 발생 | continuation-closed export와 metadata/materializer 수정; 학습 중단 |
| **Shape-compatible but semantic-incompatible state** | Direct Copy tensor는 정상이나 readout/J&F gap 큼 | Ridge→downstream loss; PE/slot order 먼저 배제 |
| **Tiny→Large information bottleneck** | Large→Tiny는 성공, Tiny→Large는 recon/readout/identity 모두 plateau | translation+replay-1/2/4, direction-specific claim, uncertainty-triggered refresh |
| **Last-Mask/replay domination** | hard event에서도 translator CI가 Last-Mask와 겹치고 replay-2 Pareto가 우세 | 과장 없이 No-Go; pair/scenario 또는 cross-architecture 필요성 재평가 |
| **Rollout/repeated-switch drift** | +1은 좋지만 +5/+20 gap과 switch-count slope가 증가 | Stage-3 rollout, target-native refresh, short replay; attention은 interaction 증거가 있을 때만 |

### 9.3 C 담당 최종 산출물 체크리스트

- [ ] Pinned `StateSpec`과 schema/version manifest
- [ ] Same-checkpoint export→inject round-trip report
- [ ] Tiny↔Large 양방향 Direct Transfer report
- [ ] Paired-state shard manifest, train-only stats, leakage audit
- [ ] Ridge/Linear sufficient-statistics fitter와 reproducible λ selection
- [ ] Component-specific Linear 및 residual MLP
- [ ] Stage 1/2/3 loss implementation과 gradient-flow unit test
- [ ] C0–C10 최소 ablation 결과
- [ ] Position checksum/conditional ablation
- [ ] B baseline과 join한 per-case raw result table
- [ ] Hard-event J&F/identity/recovery/repeated-switch 결과
- [ ] Accuracy–latency–bytes Pareto plot
- [ ] Attention/hybrid Go/No-Go memo
- [ ] D가 소비할 architecture-independent interface와 capability manifest
- [ ] 실패 사례, 방향 비대칭, 정보 병목을 포함한 최종 method table

### 9.4 이번 주 바로 할 첫 5개 행동

1. **공동:** A에게 schema dump/round-trip API, B에게 evaluator/result schema를 요청하고 interface checklist를 동결한다.
2. **C1:** `StateSpec`, validator, same-checkpoint round-trip, cross-checkpoint Direct Copy를 구현한다. **C2:** synthetic state로 loss API·gradient-flow·result join test를 병렬 작성한다.
3. **C1:** 작은 paired-state shard, train-only channel stats, Ridge sufficient statistics를 만든다. **C2:** Stage-1/2 trainer와 B evaluator integration을 완성한다.
4. **공동:** C1이 고정한 Linear checkpoint로 C2가 C0–C4 양방향을 실행하고, C1은 component intervention과 complexity 수치를 검증한다.
5. **공동:** hard-event/identity/Pareto와 residual evidence를 함께 검토해 `Linear 유지 / MLP 진입 / contract 재작업` 중 하나만 결정한다.

---

## Primary sources

- [SAM 2 paper](https://arxiv.org/html/2408.00714)
- [Official SAM 2 repository](https://github.com/facebookresearch/sam2)
- [Official SAM2VideoPredictor state management](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py)
- [Official SAM2Base memory read/write path](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py)
- [SAM 2.1 Tiny config](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml)
- [SAM 2.1 Small config](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml)
- [SAM 2.1 Large config](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml)
- [Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893) — VOS evidence가 아니라 ladder 참고
- [Mixture-of-Translators](https://arxiv.org/abs/2607.28979) — attention/mixture의 후순위 참고; CMMT 직접 증거 아님
