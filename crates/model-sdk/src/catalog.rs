use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    Transcription,
    TextGeneration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactFormat {
    Ggml,
    Gguf,
    #[serde(rename = "openvino-ir")]
    OpenVinoIr,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelProfile {
    Whisper,
    #[serde(rename = "qwen3-asr-1.7b")]
    Qwen3Asr17b,
    #[serde(rename = "qwen3-asr-0.6b")]
    Qwen3Asr06b,
    #[serde(rename = "hy-mt-1.5")]
    HyMt15,
}
