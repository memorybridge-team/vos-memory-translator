# Open research questions

These questions do not change the current Small/Large single-switch scope.

1. Do Small and Large `maskmem_features` differ mainly by a channel-wise affine
   basis change, or does a useful map require spatial context?
2. Should `obj_ptr` be translated jointly with spatial memory, or does a
   separate loss and optimizer schedule preserve identity more reliably?
3. Are target-regenerated Small and Large spatial positional encodings exactly
   equal at the pinned revision and runtime dtype?
4. Does low feature error at the switch frame predict J&F after several target
   frames, or is downstream memory-attention sensitivity strongly anisotropic?
5. How many distinct videos and switch positions are required before a linear
   map generalizes to held-out sequences rather than memorizing appearance?
6. Does translating only the bounded set of frames that target attention can
   select match translating all stored compact outputs?
7. When mask-only matches a trained translator, is the remaining gap caused by
   pointer alignment, spatial memory alignment, or target decoder calibration?
8. For repeated switching, should each handoff translate the accumulated state
   directly, or periodically refresh it from native inference to bound drift?

