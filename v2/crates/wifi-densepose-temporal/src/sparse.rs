use ruvllm_sparse_attention::{
    AttentionBackend, SparseAttentionConfig, SubquadraticSparseAttention, Tensor3,
};

use crate::{TemporalError, TemporalHeadConfig};

/// AETHER temporal head implemented with `ruvllm_sparse_attention`.
///
/// The selection rule from ADR-096 §4.4 is enforced at `forward()`
/// time: when `q_heads == kv_heads` we use `forward()` (plain MHA
/// over the sparse pattern); when they differ we use `forward_gqa()`.
/// The streaming `step()` path is staged behind a follow-up — KvCache
/// lifecycle ties to `PoseTrack` per ADR-096 §8.5 and lives on the
/// caller, not here.
pub struct SparseGqaHead {
    cfg: TemporalHeadConfig,
    attn: SubquadraticSparseAttention,
}

impl SparseGqaHead {
    pub fn new(cfg: &TemporalHeadConfig) -> Result<Self, TemporalError> {
        cfg.validate()?;

        let attn_cfg = SparseAttentionConfig {
            window: cfg.window,
            block_size: cfg.block_size,
            global_tokens: alloc_first_token(),
            causal: cfg.causal,
            use_log_stride: true,
            use_landmarks: true,
            sort_candidates: false,
        };

        let attn = SubquadraticSparseAttention::new(attn_cfg)?;
        Ok(Self {
            cfg: cfg.clone(),
            attn,
        })
    }

    pub fn cfg(&self) -> &TemporalHeadConfig {
        &self.cfg
    }

    pub fn forward(
        &self,
        q: &Tensor3,
        k: &Tensor3,
        v: &Tensor3,
    ) -> Result<Tensor3, TemporalError> {
        // ADR-096 §4.4: dispatch by GQA shape.
        if self.cfg.q_heads == self.cfg.kv_heads {
            // Pure MHA — sparse `forward` is the right path.
            Ok(self.attn.forward(q, k, v)?)
        } else {
            // GQA / MQA — kv_heads < q_heads, group share factor = q/kv.
            Ok(self.attn.forward_gqa(q, k, v)?)
        }
    }
}

/// Always treat token 0 as a global anchor — AETHER's contrastive
/// recipe (ADR-024) gives the first token a special role as the
/// "session start" reference embedding, and global tokens in the
/// sparse pattern preserve full visibility for that one position.
fn alloc_first_token() -> Vec<usize> {
    vec![0]
}
