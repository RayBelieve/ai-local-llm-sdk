use std::path::Path;

use crate::{catalog::TaskKind, engines, sdk::ModelSdk, task::TaskControl};

impl ModelSdk {
    pub fn synthesize_speech(
        &self,
        engine_id: &str,
        model_id: &str,
        text: &str,
        speed: f32,
        output: &Path,
        task: &TaskControl,
    ) -> Result<(), String> {
        task.check_cancelled()?;
        if text.trim().is_empty() {
            return Err("合成文本不能为空。".into());
        }
        if !(0.5..=2.0).contains(&speed) || !speed.is_finite() {
            return Err("语速必须在 0.5 到 2.0 之间。".into());
        }
        if output
            .extension()
            .and_then(|ext| ext.to_str())
            .is_none_or(|ext| !ext.eq_ignore_ascii_case("wav"))
        {
            return Err("输出文件必须是 WAV。".into());
        }
        if output.exists() {
            return Err("输出文件已存在。".into());
        }
        let readiness = self.check(engine_id, model_id, TaskKind::SpeechSynthesis)?;
        if !readiness.ready {
            return Err(readiness
                .reason
                .unwrap_or_else(|| "语音合成资源尚未就绪。".into()));
        }
        let engine = self.engine_path(engine_id)?.ok_or("合成引擎尚未登记。")?;
        let model = self.model_path(model_id)?.ok_or("合成模型尚未登记。")?;
        engines::onnxruntime::synthesize(self, &engine, &model, text, speed, output, task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_requests_before_loading_resources() {
        let root =
            std::env::temp_dir().join(format!("model-sdk-synthesis-test-{}", std::process::id()));
        let sdk = ModelSdk::new(&root).unwrap();
        let task = TaskControl::default();
        let output = root.join("speech.wav");
        assert!(sdk
            .synthesize_speech(
                "onnxruntime-cpu",
                crate::resources::KOKORO_MODEL_ID,
                " ",
                1.0,
                &output,
                &task
            )
            .is_err());
        assert!(sdk
            .synthesize_speech(
                "onnxruntime-cpu",
                crate::resources::KOKORO_MODEL_ID,
                "你好",
                f32::NAN,
                &output,
                &task
            )
            .is_err());
        assert!(sdk
            .synthesize_speech(
                "onnxruntime-cpu",
                crate::resources::KOKORO_MODEL_ID,
                "你好",
                1.0,
                &root.join("speech.mp3"),
                &task
            )
            .is_err());
        std::fs::write(&output, b"existing").unwrap();
        assert!(sdk
            .synthesize_speech(
                "onnxruntime-cpu",
                crate::resources::KOKORO_MODEL_ID,
                "你好",
                1.0,
                &output,
                &task
            )
            .is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"existing");
        std::fs::remove_dir_all(root).unwrap();
    }
}
