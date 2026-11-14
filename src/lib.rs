//! `jero-arch-synthetic`: Synthetic architecture recipe and deterministic token sampler.
//!
//! Implements [`ArchRecipe`] for the reference synthetic model architecture,
//! reading `mlock`ed weight tensors, updating and reading the 16-token paged KV cache,
//! dispatching `jero-kernels-rvv` primitives, and performing greedy argmax sampling
//! with lowest-token-ID tie-breaking per ADR-005, ADR-023, and ADR-024.

use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, KvCacheView, ModelMetadata, ModelWeights, QuantType,
};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{BaselineKernel, VectorKernel};

/// Synthetic architecture recipe implementation (US-01.6.3).
#[derive(Debug, Clone, Copy, Default)]
pub struct SyntheticRecipe;

impl SyntheticRecipe {
    /// Creates a new instance of the synthetic architecture recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
        #[cfg(feature = "with-kernels")]
        {
            BaselineKernel::new().dot_product_f32(lhs, rhs)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
        }
    }

    const fn bound_bias(kv_attn: f32) -> f32 {
        if kv_attn.is_finite() {
            (kv_attn.abs() % 0.25) * 0.01
        } else {
            0.0
        }
    }

    fn compute_token_features(
        self,
        token_seed: u32,
        weights: &dyn ModelWeights,
    ) -> Result<[f32; 16], K3Error> {
        #[expect(clippy::cast_precision_loss, reason = "token_seed % 17 fits in f32")]
        let base_val = ((token_seed % 17) as f32) * 0.1 + 0.5;
        let hidden = [base_val; 256];

        // 1. If `blk.0.attn_q.weight` exists in locked weights, fail closed on unsupported quants
        //    and run Q4_K_M mat-vec when shape matches.
        let mut q_proj = [base_val; 2];
        if weights.contains_tensor("blk.0.attn_q.weight") {
            let q_tensor = weights.tensor("blk.0.attn_q.weight")?;
            if !self.supports_quant(q_tensor.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    self.architecture_id(),
                    q_tensor.name(),
                    q_tensor.quant(),
                ));
            }
            if q_tensor.quant() == QuantType::Q4_K_M && q_tensor.num_elements() == 512 {
                #[cfg(feature = "with-kernels")]
                {
                    BaselineKernel::new().vec_dot_q4_k_m(&mut q_proj, &q_tensor, &hidden)?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let _ = &hidden;
                    q_proj[0] = base_val;
                    q_proj[1] = base_val * 0.5;
                }
            }
        }

        // 2. If `output_norm.weight` exists in locked weights, fail closed on unsupported quants
        //    and run RMSNorm for F32[16].
        let mut norm_out = [q_proj[0]; 16];
        if weights.contains_tensor("output_norm.weight") {
            let norm_tensor = weights.tensor("output_norm.weight")?;
            if !self.supports_quant(norm_tensor.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    self.architecture_id(),
                    norm_tensor.name(),
                    norm_tensor.quant(),
                ));
            }
            if norm_tensor.quant() == QuantType::F32 && norm_tensor.num_elements() == 16 {
                let raw = norm_tensor.as_bytes();
                let mut w_f32 = [1.0_f32; 16];
                for (i, slot) in w_f32.iter_mut().enumerate() {
                    let b = &raw[i * 4..(i + 1) * 4];
                    let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    *slot = if val.is_finite() && val.abs() < 1e4 {
                        val
                    } else {
                        1.0
                    };
                }
                let norm_in = [q_proj[0].clamp(-10.0, 10.0) + 1.0; 16];
                #[cfg(feature = "with-kernels")]
                {
                    BaselineKernel::new().rmsnorm(&mut norm_out, &norm_in, &w_f32, 1e-5)?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let _ = &w_f32;
                    norm_out = norm_in;
                }
            }
        }

        Ok(norm_out)
    }

    fn execute_raw_kv_slice(position: usize, kv_cache: &mut [f32], norm_out: &[f32; 16]) -> f32 {
        let mut kv_attn = norm_out[0];
        if !kv_cache.is_empty() {
            let kv_len = kv_cache.len();
            let stride = 16.min(kv_len);
            let start = (position.saturating_mul(stride)) % kv_len;
            for i in 0..stride {
                let idx = (start + i) % kv_len;
                kv_cache[idx] = norm_out[i % norm_out.len()];
            }

            let window_end = (start + stride).min(kv_len);
            let kv_window = &kv_cache[start..window_end];
            let q_window = &norm_out[..kv_window.len()];
            kv_attn = Self::dot_f32(kv_window, q_window);
            #[cfg(not(feature = "with-kernels"))]
            if !kv_window.is_empty() {
                let _ = kv_attn;
                kv_attn = kv_window[0];
            }
        }
        Self::bound_bias(kv_attn)
    }

    fn execute_kv_view_slot(
        kv_view: &mut KvCacheView<'_>,
        position: usize,
        norm_out: &[f32; 16],
    ) -> Result<f32, K3Error> {
        let num_layers = kv_view.geometry().num_layers;
        for layer in 0..num_layers {
            let (k_slot, v_slot) = kv_view.write_pos_range(layer, position)?;
            #[expect(clippy::cast_precision_loss, reason = "small layer count fits in f32")]
            let layer_offset = (layer as f32) * 0.01;
            for (idx, (k, v)) in k_slot.iter_mut().zip(v_slot.iter_mut()).enumerate() {
                let feat = norm_out[idx % norm_out.len()];
                *k = feat + layer_offset;
                *v = feat * 0.5 + layer_offset;
            }
        }

        let end_pos = position
            .checked_add(1)
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: kv_view.geometry().seq_capacity,
            })?;
        let (k_valid, _) = kv_view.valid_kv_range(0, end_pos)?;
        let win_len = k_valid.len().min(norm_out.len());
        let kv_attn = if win_len == 0 {
            norm_out[0]
        } else {
            Self::dot_f32(&k_valid[k_valid.len() - win_len..], &norm_out[..win_len])
        };
        Ok(Self::bound_bias(kv_attn))
    }

    fn execute_token_step(
        self,
        token_seed: u32,
        position: usize,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<f32, K3Error> {
        let norm_out = self.compute_token_features(token_seed, weights)?;
        if let Some(mut kv_view) = ctx.kv_view_mut() {
            Self::execute_kv_view_slot(&mut kv_view, position, &norm_out)
        } else {
            Ok(Self::execute_raw_kv_slice(
                position,
                ctx.kv_cache_mut(),
                &norm_out,
            ))
        }
    }
}

impl ArchRecipe for SyntheticRecipe {
    fn architecture_id(&self) -> &'static str {
        "synthetic"
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32 | QuantType::F16 | QuantType::BF16 | QuantType::Q4_K_M
        )
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if metadata.architecture() != self.architecture_id() {
            return Err(K3Error::CapabilityMismatch {
                architecture: metadata.architecture().to_owned(),
                reason: format!(
                    "SyntheticRecipe expects architecture '{}', found '{}'",
                    self.architecture_id(),
                    metadata.architecture()
                ),
            });
        }
        for desc in metadata.tensors() {
            if !self.supports_quant(desc.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    desc.name(),
                    desc.quant(),
                ));
            }
            if weights.tensor_count() > 0 {
                let view = weights.tensor(desc.name())?;
                if !self.supports_quant(view.quant()) {
                    return Err(K3Error::unsupported_tensor_quant(
                        metadata.architecture(),
                        view.name(),
                        view.quant(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let start_pos = ctx.position();
        let prompt_tokens = ctx.prompt_tokens();
        let prompt_len = prompt_tokens.len();

        // Process each prompt token sequentially starting at `start_pos` for chunked prefill parity.
        let mut bias = 0.0_f32;
        for (offset, &tok) in prompt_tokens.iter().enumerate() {
            let pos = start_pos
                .checked_add(offset)
                .ok_or(K3Error::ContextLengthExceeded {
                    requested_tokens: usize::MAX,
                    max_context_tokens: ctx.n_ctx(),
                })?;
            bias = self.execute_token_step(tok, pos, ctx, weights)?;
        }

        ctx.advance_position(prompt_len)?;

        // Produce initial deterministic logits for next token selection.
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let target_slot = if logits.len() > 9 { 9 } else { 0 };
            logits[target_slot] = 1.0 + bias;
        }

        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;

        let pos = ctx.position();
        let bias = self.execute_token_step(token, pos, ctx, weights)?;
        ctx.advance_position(1)?;

        // Produce deterministic autoregressive next-token logits.
        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
            logits[next_slot] = 2.0 + bias;
        }

        Ok(())
    }
}

/// Deterministically selects the token ID with the highest finite logit score (temperature `0.0`).
///
/// Ties are broken by selecting the lowest token ID to guarantee reproducible
/// execution per ADR-005. Non-finite logits (`NaN`, `+inf`, `-inf`) are ignored,
/// returning `None` if `logits` is empty or contains no finite scores.
#[must_use]
pub fn sample_argmax(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;

    for (i, &score) in logits.iter().enumerate() {
        if !score.is_finite() {
            continue;
        }
        // Only update on strictly greater finite score so equal scores preserve the lowest index.
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((i, score));
        }
    }

    best.and_then(|(idx, _)| u32::try_from(idx).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::{GgufFile, KvGeometry, TensorView, test_helpers::create_synthetic_gguf_v3};

    struct DummyWeights;

    impl ModelWeights for DummyWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "no tensors in dummy weights".to_owned(),
            })
        }

        fn tensor_count(&self) -> usize {
            0
        }

        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_synthetic_recipe_lifecycle() {
        let recipe = SyntheticRecipe::new();
        assert_eq!(recipe.architecture_id(), "synthetic");
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(!recipe.supports_quant(QuantType::Q8_0));

        let prompt = [10_u32, 20];
        let mut logits = [0.0_f32; 8];
        let mut kv_cache = [0.0_f32; 16];
        let weights = DummyWeights;

        let mut ctx = ExecutionContext::new(&prompt, 4, &mut logits, &mut kv_cache)
            .expect("valid execution context");

        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("prefill succeeds");
        assert_eq!(ctx.position(), 2);
        assert_eq!(sample_argmax(ctx.logits()), Some(0));

        ctx.prepare_decode(0).expect("prepare decode");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode succeeds");
        assert_eq!(ctx.position(), 3);
        assert_eq!(sample_argmax(ctx.logits()), Some(1));
    }

    #[test]
    fn test_synthetic_recipe_with_gguf_weights_and_kv_determinism_us01_6_3() {
        let tensor_specs = [
            ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
            ("output_norm.weight", QuantType::F32, &[16][..]),
        ];
        let bytes = create_synthetic_gguf_v3("synthetic", 32, &tensor_specs);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic gguf");
        let recipe = SyntheticRecipe::new();

        let prompt = [1_u32, 9, 10];
        let mut seq1 = Vec::new();
        let mut seq2 = Vec::new();

        for out_seq in [&mut seq1, &mut seq2] {
            let mut logits = [0.0_f32; 16];
            let mut kv_cache = [0.0_f32; 128];
            let mut ctx = ExecutionContext::new(&prompt, 16, &mut logits, &mut kv_cache)
                .expect("create context");

            recipe.prefill_step(&mut ctx, &gguf).expect("prefill");
            let mut tok = sample_argmax(ctx.logits()).expect("sample prefill");
            out_seq.push(tok);

            for _ in 0..3 {
                ctx.prepare_decode(tok).expect("prepare decode");
                recipe.decode_step(&mut ctx, &gguf).expect("decode");
                tok = sample_argmax(ctx.logits()).expect("sample decode");
                out_seq.push(tok);
            }

            // Confirm KV cache was populated with non-zero state
            assert!(ctx.kv_cache().iter().any(|&v| v != 0.0));
        }

        // Bit-identical token sequences across repeated runs
        assert_eq!(seq1, seq2);
        assert_eq!(seq1, vec![9, 10, 11, 12]);
    }

    #[test]
    fn test_sample_argmax_tie_breaking() {
        // Tie between index 2 and index 4: must select lowest index (2).
        let logits = [0.0_f32, 1.0, 5.0, 3.0, 5.0, 2.0];
        assert_eq!(sample_argmax(&logits), Some(2));

        // Negative zero / positive zero tie keeps lowest index.
        assert_eq!(sample_argmax(&[-0.0_f32, 0.0]), Some(0));

        // Empty logits
        assert_eq!(sample_argmax(&[]), None);

        // Single logit
        assert_eq!(sample_argmax(&[42.0]), Some(0));
    }

    #[test]
    fn test_sample_argmax_non_finite_handling() {
        // Leading NaN must never shadow finite negative or positive logits.
        assert_eq!(sample_argmax(&[f32::NAN, -5.0, -1.0]), Some(2));
        assert_eq!(
            sample_argmax(&[f32::NAN, f32::NAN, -10.0, -10.0, -12.0]),
            Some(2)
        );
        // Trailing and interior NaNs are ignored cleanly.
        assert_eq!(sample_argmax(&[-3.0, f32::NAN, -2.5, f32::NAN]), Some(2));

        // +inf and -inf are non-finite and skipped.
        assert_eq!(
            sample_argmax(&[f32::INFINITY, -5.0, f32::NEG_INFINITY, -2.0]),
            Some(3)
        );
        assert_eq!(
            sample_argmax(&[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]),
            None
        );

        // All non-finite / NaN slices return None.
        assert_eq!(sample_argmax(&[f32::NAN]), None);
        assert_eq!(sample_argmax(&[f32::NAN, f32::NAN]), None);
        assert_eq!(sample_argmax(&[f32::NEG_INFINITY]), None);
    }

    struct MismatchWeights;

    impl ModelWeights for MismatchWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "missing tensor".to_owned(),
            })
        }

        fn tensor_count(&self) -> usize {
            1
        }

        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_synthetic_recipe_validate_model_and_quant_contract() {
        let recipe = SyntheticRecipe::new();
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));
        assert!(recipe.supports_quant(QuantType::Q4_K_M));
        assert!(!recipe.supports_quant(QuantType::Q8_0));
        assert!(!recipe.supports_quant(QuantType::Q2_K));

        // Supported GGUF passes both empty and resident weights validation.
        let valid_bytes = create_synthetic_gguf_v3(
            "synthetic",
            32,
            &[
                ("blk.0.attn_q.weight", QuantType::Q4_K_M, &[256, 2][..]),
                ("output_norm.weight", QuantType::F32, &[16][..]),
            ],
        );
        let valid_gguf = GgufFile::parse(&valid_bytes).expect("parse valid gguf");
        recipe
            .validate_model(valid_gguf.metadata(), &valid_gguf)
            .expect("supported quants pass");
        recipe
            .validate_model(valid_gguf.metadata(), &DummyWeights)
            .expect("metadata-only check passes");

        // Loaded weights missing a metadata tensor fail closed.
        let missing_err = recipe
            .validate_model(valid_gguf.metadata(), &MismatchWeights)
            .expect_err("missing resident tensor must fail");
        assert!(matches!(missing_err, K3Error::TensorBoundsError { .. }));

        // Q8_0 tensor fails fail-closed admission naming tensor and Q8_0.
        let q8_bytes = create_synthetic_gguf_v3(
            "synthetic",
            32,
            &[
                ("output_norm.weight", QuantType::F32, &[16][..]),
                ("blk.0.attn_q.weight", QuantType::Q8_0, &[32, 32][..]),
            ],
        );
        let q8_gguf = GgufFile::parse(&q8_bytes).expect("parse q8 gguf");
        let q8_err = recipe
            .validate_model(q8_gguf.metadata(), &q8_gguf)
            .expect_err("Q8_0 must be rejected");
        match q8_err {
            K3Error::CapabilityMismatch {
                architecture,
                reason,
            } => {
                assert_eq!(architecture, "synthetic");
                assert!(reason.contains("blk.0.attn_q.weight"));
                assert!(reason.contains("Q8_0"));
            }
            other => panic!("expected CapabilityMismatch, got {other:?}"),
        }

        // Wrong architecture fails closed.
        let llama_bytes = create_synthetic_gguf_v3(
            "llama",
            32,
            &[("output_norm.weight", QuantType::F32, &[16][..])],
        );
        let llama_gguf = GgufFile::parse(&llama_bytes).expect("parse llama gguf");
        assert!(matches!(
            recipe.validate_model(llama_gguf.metadata(), &llama_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_synthetic_recipe_kv_cache_view_and_chunked_start_position_parity() {
        let recipe = SyntheticRecipe::new();
        let geom = KvGeometry::new(2, 2, 4, 8, Some(3)).expect("valid kv geometry");
        let req_len = geom.required_f32_len().expect("required len");

        let prompt = [3_u32, 7, 11, 13];
        let mut single_logits = [0.0_f32; 16];
        let mut single_kv = vec![0.0_f32; req_len];
        let mut single_ctx = ExecutionContext::new(&prompt, 8, &mut single_logits, &mut single_kv)
            .expect("single ctx")
            .with_kv_geometry(geom)
            .expect("attach geom");
        recipe
            .prefill_step(&mut single_ctx, &DummyWeights)
            .expect("single prefill");
        assert_eq!(single_ctx.position(), 4);

        let mut chunk_kv = vec![0.0_f32; req_len];
        let mut chunk1_logits = [0.0_f32; 16];
        {
            let mut ctx1 =
                ExecutionContext::new(&prompt[..2], 8, &mut chunk1_logits, &mut chunk_kv)
                    .expect("chunk1 ctx")
                    .with_kv_geometry(geom)
                    .expect("attach geom")
                    .with_position(0)
                    .expect("pos 0");
            recipe
                .prefill_step(&mut ctx1, &DummyWeights)
                .expect("chunk1 prefill");
            assert_eq!(ctx1.position(), 2);
        }

        let mut chunk2_logits = [0.0_f32; 16];
        let mut ctx2 = ExecutionContext::new(&prompt[2..], 8, &mut chunk2_logits, &mut chunk_kv)
            .expect("chunk2 ctx")
            .with_kv_geometry(geom)
            .expect("attach geom")
            .with_position(2)
            .expect("pos 2");
        recipe
            .prefill_step(&mut ctx2, &DummyWeights)
            .expect("chunk2 prefill");
        assert_eq!(ctx2.position(), 4);

        // Chunked prefill matches single-shot prefill in both logits and full KV arena.
        assert_eq!(single_ctx.logits(), ctx2.logits());
        assert_eq!(single_ctx.kv_cache(), ctx2.kv_cache());

        // Subsequent decode steps also produce bit-identical results.
        let tok = sample_argmax(single_ctx.logits()).expect("sample token");
        single_ctx.prepare_decode(tok).expect("prepare decode");
        ctx2.prepare_decode(tok).expect("prepare decode");
        recipe
            .decode_step(&mut single_ctx, &DummyWeights)
            .expect("single decode");
        recipe
            .decode_step(&mut ctx2, &DummyWeights)
            .expect("chunked decode");
        assert_eq!(single_ctx.position(), 5);
        assert_eq!(ctx2.position(), 5);
        assert_eq!(single_ctx.logits(), ctx2.logits());
        assert_eq!(single_ctx.kv_cache(), ctx2.kv_cache());
    }

    #[test]
    fn test_synthetic_recipe_raw_kv_chunked_start_position_and_prefill_fail_closed_q8() {
        let recipe = SyntheticRecipe::new();
        let prompt = [3_u32, 7, 11];

        let mut single_logits = [0.0_f32; 16];
        let mut single_kv = [0.0_f32; 64];
        let mut single_ctx = ExecutionContext::new(&prompt, 8, &mut single_logits, &mut single_kv)
            .expect("single raw ctx");
        recipe
            .prefill_step(&mut single_ctx, &DummyWeights)
            .expect("single raw prefill");

        let mut chunk_kv = [0.0_f32; 64];
        let mut chunk1_logits = [0.0_f32; 16];
        {
            let mut ctx1 =
                ExecutionContext::new(&prompt[..1], 8, &mut chunk1_logits, &mut chunk_kv)
                    .expect("chunk1 raw ctx")
                    .with_position(0)
                    .expect("pos 0");
            recipe
                .prefill_step(&mut ctx1, &DummyWeights)
                .expect("chunk1 raw prefill");
        }
        let mut chunk2_logits = [0.0_f32; 16];
        let mut ctx2 = ExecutionContext::new(&prompt[1..], 8, &mut chunk2_logits, &mut chunk_kv)
            .expect("chunk2 raw ctx")
            .with_position(1)
            .expect("pos 1");
        recipe
            .prefill_step(&mut ctx2, &DummyWeights)
            .expect("chunk2 raw prefill");

        assert_eq!(single_ctx.position(), ctx2.position());
        assert_eq!(single_ctx.logits(), ctx2.logits());
        assert_eq!(single_ctx.kv_cache(), ctx2.kv_cache());

        let q8_bytes = create_synthetic_gguf_v3(
            "synthetic",
            32,
            &[("blk.0.attn_q.weight", QuantType::Q8_0, &[32, 32][..])],
        );
        let q8_gguf = GgufFile::parse(&q8_bytes).expect("parse q8 gguf");
        let mut err_logits = [0.0_f32; 16];
        let mut err_kv = [0.0_f32; 64];
        let mut err_ctx =
            ExecutionContext::new(&[1], 8, &mut err_logits, &mut err_kv).expect("err ctx");
        let err = recipe
            .prefill_step(&mut err_ctx, &q8_gguf)
            .expect_err("prefill_step must fail closed on Q8_0 weights");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }
}
