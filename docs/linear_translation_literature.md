# Literature used by the linear memory translation baseline

Only primary papers and official implementations are listed. The mapping from
these works to SAM 2 memory translation is an engineering hypothesis, not a
claim made by the cited authors.

## SAM 2: Segment Anything in Images and Videos

- **Authors / year:** Nikhila Ravi et al., 2024.
- **Paper / code:** [arXiv:2408.00714](https://arxiv.org/abs/2408.00714),
  [facebookresearch/sam2](https://github.com/facebookresearch/sam2).
- **State or representation:** streaming spatial memory features, spatial and
  temporal positional information, and object pointers used by memory
  attention.
- **Mapping:** none; this is the source architecture.
- **Training data / loss:** SAM 2 pretraining is outside this repository's
  translator training. The paper describes the SA-V data engine and promptable
  segmentation objectives.
- **Direct implementation use:** follow the official predictor's compact
  per-object frame state and its memory-attention consumer. Pin the exact code
  revision.
- **Why it cannot be applied unchanged:** SAM 2 does not define cross-checkpoint
  memory transfer. Extraction, target-native PE regeneration, and injection are
  private-API extensions tested here.

## DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving

- **Authors / year:** Yuhan Liu, Yuyang Huang, Jiayi Yao, Zhuohan Gu, Kuntai Du,
  Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, Madan Musuvathi, and Esha
  Choukse, 2024 preprint; NSDI 2026 publication.
- **Paper:** [arXiv:2411.02820](https://arxiv.org/abs/2411.02820).
- **State or representation:** Transformer per-token KV caches from fine-tuned
  variants derived from the same base model.
- **Mapping:** identifies critical layers and selectively recomputes cache
  portions; it is primarily reuse/recomputation rather than a general learned
  source-to-target linear map.
- **Training data / loss:** model-serving evaluation over shared LLM contexts;
  not a VOS paired-feature regression dataset.
- **Direct implementation use:** compare state-transfer cost with replay cost,
  and record exactly what history was reused or recomputed.
- **Why it cannot be applied unchanged:** SAM 2 stores per-frame spatial memory
  and pointers rather than token-layer KV caches, and Small/Large are not merely
  fine-tunes sharing every layer shape.

## Relative Representations Enable Zero-Shot Latent Space Communication

- **Authors / venue / year:** Luca Moschella, Valentino Maiorca, Marco Fumero,
  Antonio Norelli, Francesco Locatello, and Emanuele Rodolà; ICLR 2023.
- **Paper:** [OpenReview](https://openreview.net/forum?id=SrC-nwieGJ).
- **State or representation:** encoder latent vectors represented by similarity
  to parallel anchors.
- **Mapping:** cosine-relative coordinates invariant to isometries and global
  rescaling; supports zero-shot stitching of frozen components.
- **Training data / loss:** anchor examples with known correspondence; the core
  relative transform itself is non-learned.
- **Direct implementation use:** cosine alignment is reported alongside MSE,
  and sequence-level paired samples are kept aligned.
- **Why it cannot be applied unchanged:** a dense SAM 2 memory map has spatial
  axes and target memory attention expects its native 64-channel tokens. Anchor
  representations would also change the consumer interface.

## Harnessing the Universal Geometry of Embeddings (vec2vec)

- **Authors / year:** Rishi Jha, Collin Zhang, Vitaly Shmatikov, and John X.
  Morris, 2025.
- **Paper / code:** [arXiv:2505.12540](https://arxiv.org/abs/2505.12540),
  [project and official code](https://vec2vec.github.io/).
- **State or representation:** text-encoder embeddings from different models.
- **Mapping:** an unsupervised translation through a universal latent geometry,
  without paired documents or encoder access.
- **Training data / loss:** unpaired embedding sets; geometry-preservation and
  reconstruction objectives described by the paper.
- **Direct implementation use:** motivates reporting cosine geometry and testing
  on unseen sequences rather than only reconstruction on training pairs.
- **Why it cannot be applied unchanged:** our first baseline deliberately uses
  paired state from identical video prefixes and a small affine mapper. Dense
  temporal VOS state and downstream injection differ from document embeddings.

## Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models

- **Authors / venue / year:** Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico
  Kolter, and Albert Gu; NeurIPS 2024.
- **Paper / code:** [arXiv:2408.10189](https://arxiv.org/abs/2408.10189),
  [official Phi-Mamba implementation](https://github.com/goombalab/phi-mamba).
- **State or representation:** Transformer attention mixing matrices, block
  hidden states, and final predictions aligned progressively to an SSM student.
- **Mapping:** MOHAWK performs matrix orientation, hidden-state alignment, and
  end-to-end knowledge distillation; it is not runtime KV-cache translation.
- **Training data / loss:** the paper reports C4 token subsets for the three
  distillation stages and corresponding matrix, hidden-state, and task losses.
- **Direct implementation use:** start with the narrowest representation
  alignment objective before adding downstream task loss; keep source/target
  frozen.
- **Why it cannot be applied unchanged:** Phi-Mamba changes architecture and
  trains a new student model over billions of tokens. This repository only
  trains a lightweight runtime memory adapter between frozen SAM 2 checkpoints.

## Classical alignment baselines

- **Linear least squares / ridge regression:** closed-form affine channel
  alignment is the lowest-complexity paired baseline. Ridge strength and
  normalization statistics must be stored with the translator.
- **Orthogonal Procrustes:** useful as a diagnostic if the spaces differ mostly
  by rotation/reflection, but it assumes equal dimensions and an orthogonality
  constraint. It is not a default because SAM 2 representations need not be
  isometric.
- **CKA:** a representation-similarity diagnostic, not an injectible mapping.
  It is optional after MSE and cosine metrics.
- **Low-rank affine mapping:** a parameter/latency ablation only after the full
  affine baseline is working.

## Scope note on “model stitching”

The original model-stitching literature commonly trains a light connector
between frozen or pretrained network fragments. The ICLR 2023 relative-
representation work above is the exact zero-shot stitching reference verified
for this milestone. We implement a learned affine connector because source and
target SAM 2 compact states have matched examples and the target consumer API
must retain its native tensor shape.

