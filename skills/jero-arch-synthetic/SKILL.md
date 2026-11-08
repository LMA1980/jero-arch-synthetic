---
name: jero-arch-synthetic
description: "Implement and validate the synthetic reference ArchRecipe (SyntheticRecipe), 16-token paged KV cache round-trip, and deterministic lowest-token-ID tie-breaking argmax sampler (sample_argmax) in jero-arch-synthetic."
---

# `jero-arch-synthetic` Reference Architecture Recipe Invariants

- Implement `jero_core::ArchRecipe` (`SyntheticRecipe`) with `architecture_id() == "synthetic"`, supporting `QuantType::F32`, `QuantType::F16`, `QuantType::BF16`, and `QuantType::Q4_K_M`.
- Exercise zero-copy `ModelWeights` tensor views (`blk.0.attn_q.weight` `Q4_K_M` and `output_norm.weight` `F32`), single-slot 16-token paged KV cache read/write slices, and `jero-kernels-rvv` (`vec_dot_q4_k_m`, `rmsnorm`, `dot_product_f32`) when the `with-kernels` feature is enabled.
- Maintain deterministic greedy argmax token selection (`sample_argmax`) with lowest-token-ID tie-breaking at temperature `0.0`.
- Use `usize` for slice indexing and KV cache window offsets, and `u32` for token IDs.
