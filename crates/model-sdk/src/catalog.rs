use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    Transcription,
    TextGeneration,
    ObjectDetection,
    ImageTagging,
    SpeechSynthesis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactFormat {
    Ggml,
    Gguf,
    Onnx,
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
    #[serde(rename = "hy-mt-2")]
    HyMt2,
    #[serde(rename = "yolov8-detect")]
    YoloV8Detect,
    #[serde(rename = "yolo11-detect")]
    Yolo11Detect,
    Wd14V2,
    Wd14V3,
    KokoroZh,
}
