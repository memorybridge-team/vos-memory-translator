# 참고 자료 인덱스

저장소에는 논문 원문 PDF나 자동 추출한 전문을 올리지 않고, 공식 출처 링크와
프로젝트의 분석 문서만 유지합니다. 필요한 원문은 각 공식 사이트에서 내려받아
Git에서 제외되는 `references/papers/`에 보관할 수 있습니다.

## 핵심 논문

- *SAM 2: Segment Anything in Images and Videos*, ICLR 2025
  - [Official repository](https://github.com/facebookresearch/sam2)
  - [arXiv](https://arxiv.org/abs/2408.00714)
- *Cross-Model KV Cache Transfer in LLM Families*, arXiv:2608.03893
  - [arXiv](https://arxiv.org/abs/2608.03893)
- *Efficient Video Object Segmentation and Tracking with Recurrent Dynamic
  Submodel*, CVPR 2026
  - [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Efficient_Video_Object_Segmentation_and_Tracking_with_Recurrent_Dynamic_Submodel_CVPR_2026_paper.html)

## 프로젝트 분석

- [KV cache 방법을 SAM 2에 적용한 구현 보고서](../docs/design/CROSS_MODEL_KV_TO_SAM2_IMPLEMENTATION_REPORT.md)
- [Translator·방법론 설계 보고서](../docs/design/C_TRANSLATOR_METHOD_REPORT.md)
- [현재 정식 실험 계획](../docs/experimental_plan.md)

원문에서 확인한 사실, 아직 검증하지 않은 가설, 제한적인 pilot 결과를 문서에서
구분합니다. 수식·표·그림은 이 저장소의 추출본이 아니라 공식 원문을 기준으로
확인합니다.
