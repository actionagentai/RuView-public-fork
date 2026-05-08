// AETHER temporal head over CSI feature windows (ADR-096).
//
// Wraps `ruvllm_sparse_attention::SubquadraticSparseAttention` so AETHER
// callers in `wifi-densepose-train` and `wifi-densepose-signal` can swap
// dense MHA for sparse-GQA without touching the contrastive recipe.
//
// Status: scaffolding for ADR-096 §4.3. Sparse backend is functional;
// the dense back-compat backend is a follow-up (Phase 2 of the roadmap
// in #513). Streaming `step()` lands once the per-track KvCache lifecycle
// (ADR-096 §8.5) is finalized.

pub mod config;
pub mod error;
pub mod sparse;

pub use config::{TemporalBackendKind, TemporalHeadConfig};
pub use error::TemporalError;
pub use sparse::SparseGqaHead;

// Re-export the upstream Tensor3 so callers don't need a direct
// `ruvllm_sparse_attention` dep.
pub use ruvllm_sparse_attention::Tensor3;

/// Thin facade so callers can pick a backend by name.
///
/// Today only `SparseGqa` is implemented; `Dense` is reserved per
/// ADR-096 §4.4 and returns `TemporalError::DenseBackendNotImplemented`
/// until the back-compat path lands.
pub enum AetherTemporalHead {
    SparseGqa(SparseGqaHead),
    Dense, // placeholder; ADR-096 §4.4 selection rule
}

impl AetherTemporalHead {
    pub fn new(cfg: &TemporalHeadConfig) -> Result<Self, TemporalError> {
        match cfg.backend {
            TemporalBackendKind::SparseGqa => {
                Ok(AetherTemporalHead::SparseGqa(SparseGqaHead::new(cfg)?))
            }
            TemporalBackendKind::Dense => Err(TemporalError::DenseBackendNotImplemented),
        }
    }

    /// Window-level prefill. Returns the per-token attention output as
    /// a Tensor3 of shape (window, q_heads, head_dim). Pooling to a
    /// single embedding is the caller's responsibility — different
    /// AETHER consumers use different pool ops (mean for re-ID,
    /// last-token for streaming).
    pub fn forward(
        &self,
        q: &Tensor3,
        k: &Tensor3,
        v: &Tensor3,
    ) -> Result<Tensor3, TemporalError> {
        match self {
            AetherTemporalHead::SparseGqa(h) => h.forward(q, k, v),
            AetherTemporalHead::Dense => Err(TemporalError::DenseBackendNotImplemented),
        }
    }
}
