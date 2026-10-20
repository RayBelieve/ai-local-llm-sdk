use std::{
    fs::File,
    io::Read,
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::catalog::{ArtifactFormat, ModelProfile, TaskKind};

pub const ENGINE_VERSION: &str = "v0.8.33";
pub const LLAMA_ENGINE_VERSION: &str = "b9500";
pub const ONNXRUNTIME_ENGINE_VERSION: &str = "1.30.0";
pub const HY_MT_MODEL_ID: &str = "hy-mt-2-1.8b-q4";
pub const OPENVINO_ENGINE_ID: &str = "openvino-cpu";
pub const OPENVINO_MODEL_ID: &str = "qwen3-asr-0.6b-int8";
pub const KOKORO_MODEL_ID: &str = "kokoro-82m-v1.1-zh";
pub const KOKORO_REVISION: &str = "6cc0f0d2ebe369a68b0df87c2b65c1af8c0ac3e3";
pub const KOKORO_FILES: &[(&str, &str, u64, &str)] = &[
    (
        "model.onnx",
        "onnx/model.onnx",
        339_369_442,
        "94b973941b1852754f979be5d5e20be666d5c81d9bb886b88ae1dc85c9b895ca",
    ),
    (
        "tokenizer.json",
        "tokenizer.json",
        4_944,
        "5715a60b09d5e4b9074435d68c6ccd5675b9d48b220e109fdea3cda681e23d15",
    ),
    (
        "zf_001.bin",
        "voices/zf_001.bin",
        522_240,
        "0a89ec12bb93fb9c74077924daf02568baad64e1f869389f5aaee01a386035f8",
    ),
];
pub const OPENVINO_MODEL_FILES: &[&str] = &[
    "audio_encoder_model.bin",
    "audio_encoder_model.xml",
    "decoder_model.bin",
    "decoder_model.xml",
    "thinker_embeddings_model.bin",
    "thinker_embeddings_model.xml",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
];
pub const OPENVINO_MODEL_HASHES: &[(&str, &str)] = &[
    (
        "audio_encoder_model.bin",
        "d892464d9b6986719dd6e5c3962b880a2708d874c2c9bdead8958581be2dacb9",
    ),
    (
        "decoder_model.bin",
        "cc4363c401f5faf41e2bfcb4aea80c72144b8ea66d13ca5ca62cf49421a25778",
    ),
    (
        "thinker_embeddings_model.bin",
        "a7818fcbd77240fb8705bc47c2a15da98498056cdd419742b7685719b5dc2a44",
    ),
];
pub const ENGINE_BASE: &str = "https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.33";
pub const LLAMA_ENGINE_BASE: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b9500";
const WHISPER_REPO: &str = "ggerganov/whisper.cpp";

pub struct LlamaAsset {
    pub file: &'static str,
    pub size: u64,
}

pub struct LlamaVariant {
    pub id: &'static str,
    pub label: &'static str,
    pub assets: &'static [LlamaAsset],
}

// shortcut: upstream release ZIPs have pinned sizes but no recorded SHA-256; pin digests before adding new versions.
pub const LLAMA_VARIANTS: &[LlamaVariant] = &[
    LlamaVariant {
        id: "llama-cpu",
        label: "CPU",
        assets: &[LlamaAsset {
            file: "llama-b9500-bin-win-cpu-x64.zip",
            size: 16_906_094,
        }],
    },
    LlamaVariant {
        id: "llama-vulkan",
        label: "Vulkan",
        assets: &[LlamaAsset {
            file: "llama-b9500-bin-win-vulkan-x64.zip",
            size: 33_878_631,
        }],
    },
    LlamaVariant {
        id: "llama-cuda12",
        label: "CUDA 12.4",
        assets: &[
            LlamaAsset {
                file: "llama-b9500-bin-win-cuda-12.4-x64.zip",
                size: 261_143_138,
            },
            LlamaAsset {
                file: "cudart-llama-bin-win-cuda-12.4-x64.zip",
                size: 391_443_627,
            },
        ],
    },
    LlamaVariant {
        id: "llama-cuda13",
        label: "CUDA 13.3",
        assets: &[
            LlamaAsset {
                file: "llama-b9500-bin-win-cuda-13.3-x64.zip",
                size: 159_076_541,
            },
            LlamaAsset {
                file: "cudart-llama-bin-win-cuda-13.3-x64.zip",
                size: 390_970_417,
            },
        ],
    },
];

pub fn llama_variant(id: &str) -> Result<&'static LlamaVariant, String> {
    LLAMA_VARIANTS
        .iter()
        .find(|variant| variant.id == id)
        .ok_or_else(|| format!("未知 llama.cpp 运行时版本: {id}"))
}

pub struct EngineSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub asset: &'static str,
    pub gpu_backend: &'static str,
    pub size_mb: u64,
    pub requires_avx2: bool,
    pub min_nvidia_driver: Option<u32>,
    pub min_compute_capability: Option<u32>,
}

pub const ENGINES: &[EngineSpec] = &[
    EngineSpec {
        id: "crispasr-vulkan",
        label: "Vulkan",
        asset: "crispasr-windows-x86_64-vulkan.zip",
        gpu_backend: "vulkan",
        size_mb: 35,
        requires_avx2: false,
        min_nvidia_driver: None,
        min_compute_capability: None,
    },
    EngineSpec {
        id: "crispasr-cuda12",
        label: "CUDA 12",
        asset: "crispasr-windows-x86_64-cuda.zip",
        gpu_backend: "cuda",
        size_mb: 692,
        requires_avx2: false,
        min_nvidia_driver: Some(528),
        min_compute_capability: Some(50),
    },
    EngineSpec {
        id: "crispasr-cuda13",
        label: "CUDA 13",
        asset: "crispasr-windows-x86_64-cuda13.zip",
        gpu_backend: "cuda",
        size_mb: 486,
        requires_avx2: false,
        min_nvidia_driver: Some(580),
        min_compute_capability: Some(75),
    },
    EngineSpec {
        id: "crispasr-cpu",
        label: "CPU",
        asset: "crispasr-windows-x86_64-cpu.zip",
        gpu_backend: "cpu",
        size_mb: 8,
        requires_avx2: true,
        min_nvidia_driver: None,
        min_compute_capability: None,
    },
    EngineSpec {
        id: "crispasr-cpu-legacy",
        label: "CPU 兼容版",
        asset: "crispasr-windows-x86_64-cpu-legacy.zip",
        gpu_backend: "cpu",
        size_mb: 8,
        requires_avx2: false,
        min_nvidia_driver: None,
        min_compute_capability: None,
    },
];

pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub file: &'static str,
    pub repo: &'static str,
    pub format: ArtifactFormat,
    pub profile: ModelProfile,
    pub task: TaskKind,
    pub size: u64,
    pub sha256: &'static str,
}

// Sizes and LFS SHA-256 values come from each Hugging Face model repository.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "tiny",
        label: "Whisper Tiny",
        file: "ggml-tiny.bin",
        repo: WHISPER_REPO,
        format: ArtifactFormat::Ggml,
        profile: ModelProfile::Whisper,
        task: TaskKind::Transcription,
        size: 77_691_713,
        sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    },
    ModelSpec {
        id: "base",
        label: "Whisper Base",
        file: "ggml-base.bin",
        repo: WHISPER_REPO,
        format: ArtifactFormat::Ggml,
        profile: ModelProfile::Whisper,
        task: TaskKind::Transcription,
        size: 147_951_465,
        sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
    },
    ModelSpec {
        id: "small",
        label: "Whisper Small",
        file: "ggml-small.bin",
        repo: WHISPER_REPO,
        format: ArtifactFormat::Ggml,
        profile: ModelProfile::Whisper,
        task: TaskKind::Transcription,
        size: 487_601_967,
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    },
    ModelSpec {
        id: "medium",
        label: "Whisper Medium",
        file: "ggml-medium.bin",
        repo: WHISPER_REPO,
        format: ArtifactFormat::Ggml,
        profile: ModelProfile::Whisper,
        task: TaskKind::Transcription,
        size: 1_533_763_059,
        sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
    },
    ModelSpec {
        id: "turbo",
        label: "Whisper Large V3 Turbo",
        file: "ggml-large-v3-turbo.bin",
        repo: WHISPER_REPO,
        format: ArtifactFormat::Ggml,
        profile: ModelProfile::Whisper,
        task: TaskKind::Transcription,
        size: 1_624_555_275,
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    },
    ModelSpec {
        id: "large-v2",
        label: "Whisper Large V2",
        file: "ggml-large-v2.bin",
        repo: WHISPER_REPO,
        format: ArtifactFormat::Ggml,
        profile: ModelProfile::Whisper,
        task: TaskKind::Transcription,
        size: 3_094_623_691,
        sha256: "9a423fe4d40c82774b6af34115b8b935f34152246eb19e80e376071d3f999487",
    },
    ModelSpec {
        id: "qwen3-1.7b-q4",
        label: "Qwen3-ASR 1.7B Q4",
        file: "qwen3-asr-1.7b-q4_k.gguf",
        repo: "cstr/qwen3-asr-1.7b-GGUF",
        format: ArtifactFormat::Gguf,
        profile: ModelProfile::Qwen3Asr17b,
        task: TaskKind::Transcription,
        size: 1_490_915_200,
        sha256: "ec197cef7ccc589fdcae1becc3f4a3de119d0a41e790b898b519b1a048dad8d4",
    },
    ModelSpec {
        id: "qwen3-1.7b-q8",
        label: "Qwen3-ASR 1.7B Q8",
        file: "qwen3-asr-1.7b-q8_0.gguf",
        repo: "cstr/qwen3-asr-1.7b-GGUF",
        format: ArtifactFormat::Gguf,
        profile: ModelProfile::Qwen3Asr17b,
        task: TaskKind::Transcription,
        size: 2_506_723_200,
        sha256: "9851ab996591a2d0cb0efb216002764b509c86bd40c95e613d7b65b8e69c8a6e",
    },
    ModelSpec {
        id: "qwen3-1.7b-ja-q4",
        label: "Qwen3-ASR 1.7B 日语动漫 Q4",
        file: "qwen3-asr-1.7b-ja-anime-q4_k.gguf",
        repo: "cstr/qwen3-asr-1.7b-ja-anime-GGUF",
        format: ArtifactFormat::Gguf,
        profile: ModelProfile::Qwen3Asr17b,
        task: TaskKind::Transcription,
        size: 1_490_917_376,
        sha256: "da00cec556885a60d47234b4def0dbdd08394df4d4df13866d7cd9726c23ccd9",
    },
    ModelSpec {
        id: "qwen3-1.7b-ja-q8",
        label: "Qwen3-ASR 1.7B 日语动漫 Q8",
        file: "qwen3-asr-1.7b-ja-anime-q8_0.gguf",
        repo: "cstr/qwen3-asr-1.7b-ja-anime-GGUF",
        format: ArtifactFormat::Gguf,
        profile: ModelProfile::Qwen3Asr17b,
        task: TaskKind::Transcription,
        size: 2_506_725_376,
        sha256: "f173ceddb15b59177575a101491862bd362cd85f2e7a6bac811fbc53411596a1",
    },
    ModelSpec {
        id: HY_MT_MODEL_ID,
        label: "HY-MT2 1.8B Q4_K_M",
        file: "Hy-MT2-1.8B-Q4_K_M.gguf",
        repo: "tencent/Hy-MT2-1.8B-GGUF",
        format: ArtifactFormat::Gguf,
        profile: ModelProfile::HyMt2,
        task: TaskKind::TextGeneration,
        size: 1_133_080_448,
        sha256: "dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699",
    },
    ModelSpec {
        id: "hy-mt-2-1.8b-q6",
        label: "HY-MT2 1.8B Q6_K",
        file: "Hy-MT2-1.8B-Q6_K.gguf",
        repo: "tencent/Hy-MT2-1.8B-GGUF",
        format: ArtifactFormat::Gguf,
        profile: ModelProfile::HyMt2,
        task: TaskKind::TextGeneration,
        size: 1_474_785_120,
        sha256: "d98fe604dec1f28f58f80d7d560f7177e584d3b8e5835862687660e5ff97cb40",
    },
    ModelSpec {
        id: "hy-mt-2-1.8b-q8",
        label: "HY-MT2 1.8B Q8_0",
        file: "Hy-MT2-1.8B-Q8_0.gguf",
        repo: "tencent/Hy-MT2-1.8B-GGUF",
        format: ArtifactFormat::Gguf,
        profile: ModelProfile::HyMt2,
        task: TaskKind::TextGeneration,
        size: 1_908_528_192,
        sha256: "5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4",
    },
    ModelSpec {
        id: "yolov8n-det-fp32",
        label: "YOLOv8 Nano Detection FP32",
        file: "yolov8n-det-fp32.onnx",
        repo: "EdgeFirst/yolov8-det",
        format: ArtifactFormat::Onnx,
        profile: ModelProfile::YoloV8Detect,
        task: TaskKind::ObjectDetection,
        size: 12_873_047,
        sha256: "2f6148e3dca2493865f48c88186f2127e5ffe292d83b6d3df783d551def49a66",
    },
    ModelSpec {
        id: "yolov8s-det-fp32",
        label: "YOLOv8 Small Detection FP32",
        file: "yolov8s-det-fp32.onnx",
        repo: "EdgeFirst/yolov8-det",
        format: ArtifactFormat::Onnx,
        profile: ModelProfile::YoloV8Detect,
        task: TaskKind::ObjectDetection,
        size: 44_891_767,
        sha256: "ce0bc0ae36df2fa5b933dc8a69d9e0edb7085a4472160cd1d4590edd7677960a",
    },
    ModelSpec {
        id: "yolo11n-det-fp32",
        label: "YOLO11 Nano Detection FP32",
        file: "yolo11n-det-fp32.onnx",
        repo: "EdgeFirst/yolo11-det",
        format: ArtifactFormat::Onnx,
        profile: ModelProfile::Yolo11Detect,
        task: TaskKind::ObjectDetection,
        size: 10_756_872,
        sha256: "40e2f56bf626fab2d4851148f7516c6e36f852c8a42d77fd2439c4505d8f04e7",
    },
    ModelSpec {
        id: "yolo11s-det-fp32",
        label: "YOLO11 Small Detection FP32",
        file: "yolo11s-det-fp32.onnx",
        repo: "EdgeFirst/yolo11-det",
        format: ArtifactFormat::Onnx,
        profile: ModelProfile::Yolo11Detect,
        task: TaskKind::ObjectDetection,
        size: 38_067_151,
        sha256: "0dd27fa741bce2dbc0d26d0dbfb5ae7dcb851d3dbcd91298eba1cef9464d4f1b",
    },
];

pub struct Wd14Spec {
    pub id: &'static str,
    pub label: &'static str,
    pub revision: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub csv_size: u64,
    pub csv_sha256: &'static str,
    pub profile: ModelProfile,
}

// Fixed Hugging Face revisions and SHA-256 digests for both files in each model.
pub const WD14_MODELS: &[Wd14Spec] = &[
    Wd14Spec {
        id: "wd-v1-4-moat-tagger-v2",
        label: "WD 1.4 MoAT v2",
        revision: "8452cddf280b952281b6e102411c50e981cb2908",
        size: 326_197_340,
        sha256: "b8cef913be4c9e8d93f9f903e74271416502ce0b4b04df0ff1e2f00df488aa03",
        csv_size: 253_906,
        csv_sha256: "8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368",
        profile: ModelProfile::Wd14V2,
    },
    Wd14Spec {
        id: "wd-v1-4-swinv2-tagger-v2",
        label: "WD 1.4 SwinV2 v2",
        revision: "cdb0c7fdc70646f0af29c6f80f8df564344a69b6",
        size: 455_409_152,
        sha256: "04ec04fdf7db74b4fed7f4b52f52e04dec4dbad9e4d88d2d178f334079a29fde",
        csv_size: 253_906,
        csv_sha256: "8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368",
        profile: ModelProfile::Wd14V2,
    },
    Wd14Spec {
        id: "wd-v1-4-convnext-tagger-v2",
        label: "WD 1.4 ConvNeXt v2",
        revision: "4b34d1b07bdd8e95494072648960b8a6adcbc0ff",
        size: 387_820_405,
        sha256: "71f06ecb7b9df81d8f271da4d43997ea2ed363cdac29aa64fcb256c9631e656a",
        csv_size: 253_906,
        csv_sha256: "8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368",
        profile: ModelProfile::Wd14V2,
    },
    Wd14Spec {
        id: "wd-v1-4-vit-tagger-v2",
        label: "WD 1.4 ViT v2",
        revision: "1f3f3e8ae769634e31e1ef696df11ec37493e4f2",
        size: 373_287_959,
        sha256: "8a21cadd1f88a095094cafbffe3028c3cc3d97f4d58c54344c5994bcf48e24ac",
        csv_size: 253_906,
        csv_sha256: "8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368",
        profile: ModelProfile::Wd14V2,
    },
    Wd14Spec {
        id: "wd-v1-4-convnextv2-tagger-v2",
        label: "WD 1.4 ConvNeXtV2 v2",
        revision: "dbd4dbe553ee51feb3bc745b614fb762080e3912",
        size: 388_479_878,
        sha256: "e91daa19cd9e8725125b7d70702d1560855fb687f8d8c4218eddaa821f41834a",
        csv_size: 253_906,
        csv_sha256: "8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368",
        profile: ModelProfile::Wd14V2,
    },
    Wd14Spec {
        id: "wd-swinv2-tagger-v3",
        label: "WD SwinV2 v3",
        revision: "627aef95638667ddcaa3ac8ae625e88ea5b02f51",
        size: 467_460_978,
        sha256: "e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1",
        csv_size: 308_468,
        csv_sha256: "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
        profile: ModelProfile::Wd14V3,
    },
    Wd14Spec {
        id: "wd-convnext-tagger-v3",
        label: "WD ConvNeXt v3",
        revision: "d39e46de298d27340111b64965e20b8185c407e6",
        size: 394_990_732,
        sha256: "1b8a7abf13d9b8368267df47501d523789c4aeae66b2296ad98483239dfa32eb",
        csv_size: 308_468,
        csv_sha256: "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
        profile: ModelProfile::Wd14V3,
    },
    Wd14Spec {
        id: "wd-vit-tagger-v3",
        label: "WD ViT v3",
        revision: "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f",
        size: 378_536_310,
        sha256: "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6",
        csv_size: 308_468,
        csv_sha256: "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
        profile: ModelProfile::Wd14V3,
    },
    Wd14Spec {
        id: "wd-eva02-large-tagger-v3",
        label: "WD EVA02 Large v3",
        revision: "b25b82a03f7282e41aa2f257a52c7583b710bd1c",
        size: 1_260_435_999,
        sha256: "9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc",
        csv_size: 308_468,
        csv_sha256: "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
        profile: ModelProfile::Wd14V3,
    },
];

pub fn wd14_spec(id: &str) -> Option<&'static Wd14Spec> {
    WD14_MODELS.iter().find(|spec| spec.id == id)
}

pub fn verify_kokoro_model(path: &Path) -> Result<(), String> {
    if !path.is_dir() {
        return Err(
            "请选择包含 model.onnx、tokenizer.json 和 zf_001.bin 的 Kokoro 模型目录。".into(),
        );
    }
    for (name, _, size, hash) in KOKORO_FILES {
        let file = path.join(name);
        if file
            .metadata()
            .map_err(|_| format!("模型缺少 {name}"))?
            .len()
            != *size
        {
            return Err(format!("模型文件大小不符: {name}"));
        }
        crate::install::verify_sha256(&file, hash)?;
    }
    Ok(())
}

pub fn wd14_url(spec: &Wd14Spec, file: &str, mirror: bool) -> String {
    let host = if mirror {
        "https://hf-mirror.com"
    } else {
        "https://huggingface.co"
    };
    format!(
        "{host}/SmilingWolf/{}/resolve/{}/{file}",
        spec.id, spec.revision
    )
}

pub fn verify_wd14_model(path: &Path, spec: &Wd14Spec) -> Result<(), String> {
    if !path.is_dir() {
        return Err("请选择包含 model.onnx 与 selected_tags.csv 的 WD-14 模型目录。".into());
    }
    for (name, size, hash) in [
        ("model.onnx", spec.size, spec.sha256),
        ("selected_tags.csv", spec.csv_size, spec.csv_sha256),
    ] {
        let file = path.join(name);
        if file
            .metadata()
            .map_err(|_| format!("模型缺少 {name}"))?
            .len()
            != size
        {
            return Err(format!("模型文件大小不符: {name}"));
        }
        crate::install::verify_sha256(&file, hash)?;
    }
    Ok(())
}

pub fn engine_spec(id: &str) -> Result<&'static EngineSpec, String> {
    ENGINES
        .iter()
        .find(|spec| spec.id == id)
        .ok_or_else(|| format!("未知推理引擎版本: {id}"))
}

pub fn model_spec(id: &str) -> Result<&'static ModelSpec, String> {
    MODELS
        .iter()
        .find(|spec| spec.id == id)
        .ok_or_else(|| format!("未知模型: {id}"))
}

pub fn model_url(spec: &ModelSpec, mirror: bool) -> String {
    let host = if mirror {
        "https://hf-mirror.com"
    } else {
        "https://huggingface.co"
    };
    let revision = match spec.profile {
        ModelProfile::YoloV8Detect => "7fb4413528af0e22c489e7b85839df6e21fa221f",
        ModelProfile::Yolo11Detect => "6a09aa9c4b5e98370b84cacc97e79b2a9a7ae5cb",
        _ => "main",
    };
    let subdir = if spec.format == ArtifactFormat::Onnx {
        "onnx/"
    } else {
        ""
    };
    format!(
        "{host}/{}/resolve/{revision}/{subdir}{}",
        spec.repo, spec.file
    )
}

pub fn verify_model(path: &Path, spec: &ModelSpec) -> Result<(), String> {
    let size = path.metadata().map_err(|error| error.to_string())?.len();
    if size != spec.size {
        return Err(format!(
            "模型大小不符: {size} 字节，应为 {} 字节",
            spec.size
        ));
    }
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    if format!("{:x}", hash.finalize()) != spec.sha256 {
        return Err("模型 SHA-256 校验失败。".into());
    }
    Ok(())
}

pub fn verify_openvino_model(path: &Path) -> Result<(), String> {
    if !path.is_dir() {
        return Err("请选择 OpenVINO 模型目录。".into());
    }
    for name in OPENVINO_MODEL_FILES {
        let file = path.join(name);
        let metadata = file.metadata().map_err(|_| format!("模型缺少 {name}"))?;
        if !metadata.is_file() || metadata.len() < 100 {
            return Err(format!("模型文件无效: {name}"));
        }
    }
    for (name, _) in OPENVINO_MODEL_HASHES {
        if path
            .join(name)
            .metadata()
            .is_ok_and(|value| value.len() < 10_000_000)
        {
            return Err(format!("模型权重不完整: {name}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    #[test]
    fn unknown_resource_ids_are_rejected() {
        assert!(engine_spec("openvino").is_err());
        assert!(model_spec("qwen3").is_err());
    }

    #[test]
    fn hy_mt2_gguf_catalog_lists_quantizations() {
        let specs: Vec<_> = MODELS
            .iter()
            .filter(|spec| spec.task == TaskKind::TextGeneration)
            .collect();
        assert_eq!(specs.len(), 3);
        for spec in specs {
            assert_eq!(spec.profile, ModelProfile::HyMt2);
            assert_eq!(spec.format, ArtifactFormat::Gguf);
            assert_eq!(spec.repo, "tencent/Hy-MT2-1.8B-GGUF");
            assert!(spec.file.starts_with("Hy-MT2-1.8B-"));
            assert!(spec.file.ends_with(".gguf"));
            assert_eq!(spec.sha256.len(), 64);
            assert!(model_url(spec, false).contains(
                "huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/"
            ));
            assert!(model_url(spec, true).starts_with("https://hf-mirror.com/"));
        }
        assert_eq!(HY_MT_MODEL_ID, "hy-mt-2-1.8b-q4");
        assert_eq!(
            model_spec(HY_MT_MODEL_ID).unwrap().file,
            "Hy-MT2-1.8B-Q4_K_M.gguf"
        );
        assert_eq!(
            model_spec("hy-mt-2-1.8b-q6").unwrap().file,
            "Hy-MT2-1.8B-Q6_K.gguf"
        );
        assert_eq!(
            model_spec("hy-mt-2-1.8b-q8").unwrap().file,
            "Hy-MT2-1.8B-Q8_0.gguf"
        );
    }

    #[test]
    fn qwen3_gguf_catalog_uses_matching_repositories() {
        for id in [
            "qwen3-1.7b-q4",
            "qwen3-1.7b-q8",
            "qwen3-1.7b-ja-q4",
            "qwen3-1.7b-ja-q8",
        ] {
            let spec = model_spec(id).unwrap();
            assert_eq!(spec.format, ArtifactFormat::Gguf);
            assert_eq!(spec.profile, ModelProfile::Qwen3Asr17b);
            assert!(spec.file.ends_with(".gguf"));
            assert_eq!(spec.sha256.len(), 64);
            assert!(model_url(spec, false).contains(spec.repo));
            assert!(model_url(spec, true).starts_with("https://hf-mirror.com/"));
        }
    }

    #[test]
    fn yolo_downloads_use_pinned_onnx_revisions() {
        for id in [
            "yolov8n-det-fp32",
            "yolov8s-det-fp32",
            "yolo11n-det-fp32",
            "yolo11s-det-fp32",
        ] {
            let spec = model_spec(id).unwrap();
            let url = model_url(spec, false);
            assert!(url.contains("/resolve/"));
            assert!(url.contains("/onnx/"));
            assert!(url.ends_with(spec.file));
            assert_eq!(spec.task, TaskKind::ObjectDetection);
        }
    }

    #[test]
    fn wd14_catalog_uses_pinned_pair_of_files() {
        assert_eq!(WD14_MODELS.len(), 9);
        for spec in WD14_MODELS {
            for file in ["model.onnx", "selected_tags.csv"] {
                assert!(wd14_url(spec, file, false).contains(spec.revision));
                assert!(wd14_url(spec, file, true).starts_with("https://hf-mirror.com/"));
            }
            assert_eq!(spec.sha256.len(), 64);
            assert_eq!(spec.csv_sha256.len(), 64);
        }
        let spec = wd14_spec("wd-swinv2-tagger-v3").unwrap();
        let dir = crate::test_support::scratch_path("wd14-test");
        assert!(verify_wd14_model(&dir, spec)
            .unwrap_err()
            .contains("model.onnx"));
        fs::write(dir.join("model.onnx"), b"wrong").unwrap();
        assert!(verify_wd14_model(&dir, spec)
            .unwrap_err()
            .contains("大小不符"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn wd14_offline_directory_requires_both_matching_files() {
        let spec = Wd14Spec {
            id: "test",
            label: "test",
            revision: "test",
            profile: ModelProfile::Wd14V3,
            size: 1,
            sha256: "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
            csv_size: 1,
            csv_sha256: "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d",
        };
        let dir = crate::test_support::scratch_path("wd14-pair-test");
        fs::write(dir.join("model.onnx"), b"a").unwrap();
        assert!(verify_wd14_model(&dir, &spec).is_err());
        fs::write(dir.join("selected_tags.csv"), b"b").unwrap();
        verify_wd14_model(&dir, &spec).unwrap();
        fs::write(dir.join("selected_tags.csv"), b"c").unwrap();
        assert!(verify_wd14_model(&dir, &spec).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
