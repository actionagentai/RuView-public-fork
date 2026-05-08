use thiserror::Error;

#[derive(Debug, Error)]
pub enum TemporalError {
    #[error("temporal head config invalid: {0}")]
    InvalidConfig(&'static str),

    #[error("dense MHA backend not implemented yet (ADR-096 §4.4 follow-up)")]
    DenseBackendNotImplemented,

    #[error("sparse attention kernel error: {0}")]
    Kernel(String),
}

impl From<ruvllm_sparse_attention::AttentionError> for TemporalError {
    fn from(e: ruvllm_sparse_attention::AttentionError) -> Self {
        TemporalError::Kernel(format!("{e}"))
    }
}
