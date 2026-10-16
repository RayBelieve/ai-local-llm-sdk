use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{catalog::TaskKind, engines, sdk::ModelSdk, task::TaskControl};

#[derive(Clone, Debug)]
pub struct DetectionOptions {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub max_detections: usize,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            max_detections: 100,
        }
    }
}

impl DetectionOptions {
    fn validate(&self) -> Result<(), String> {
        if !self.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.confidence_threshold)
            || !self.iou_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.iou_threshold)
            || !(1..=1000).contains(&self.max_detections)
        {
            return Err("检测阈值或最大目标数无效。".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Detection {
    pub class_id: usize,
    pub label: String,
    pub confidence: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionResult {
    pub image_width: u32,
    pub image_height: u32,
    pub detections: Vec<Detection>,
}

impl ModelSdk {
    pub fn detect_objects(
        &self,
        engine_id: &str,
        model_id: &str,
        image: &Path,
        options: &DetectionOptions,
        task: &TaskControl,
    ) -> Result<DetectionResult, String> {
        options.validate()?;
        task.check_cancelled()?;
        if !image.is_file() {
            return Err(format!("找不到图片: {}", image.display()));
        }
        let readiness = self.check(engine_id, model_id, TaskKind::ObjectDetection)?;
        if !readiness.ready {
            return Err(readiness
                .reason
                .unwrap_or_else(|| "检测资源尚未就绪。".into()));
        }
        let engine = self.engine_path(engine_id)?.ok_or("检测引擎尚未登记。")?;
        let model = self.model_path(model_id)?.ok_or("检测模型尚未登记。")?;
        engines::get(engine_id)?.detect(self, &engine, &model, image, options, task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_detection_options() {
        let mut options = DetectionOptions::default();
        assert!(options.validate().is_ok());
        options.confidence_threshold = f32::NAN;
        assert!(options.validate().is_err());
        options.confidence_threshold = 0.25;
        options.max_detections = 0;
        assert!(options.validate().is_err());
    }
}
