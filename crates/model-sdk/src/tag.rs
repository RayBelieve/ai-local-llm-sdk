use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{catalog::TaskKind, engines, sdk::ModelSdk, task::TaskControl};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TagScore {
    pub name: String,
    pub category: u8,
    pub score: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TagResult {
    pub ratings: Vec<TagScore>,
    pub tags: Vec<TagScore>,
}

impl ModelSdk {
    pub fn tag_image(
        &self,
        engine_id: &str,
        model_id: &str,
        image: &Path,
        task: &TaskControl,
    ) -> Result<TagResult, String> {
        task.check_cancelled()?;
        if !image.is_file() {
            return Err(format!("找不到图片: {}", image.display()));
        }
        let readiness = self.check(engine_id, model_id, TaskKind::ImageTagging)?;
        if !readiness.ready {
            return Err(readiness
                .reason
                .unwrap_or_else(|| "打标资源尚未就绪。".into()));
        }
        let engine = self.engine_path(engine_id)?.ok_or("打标引擎尚未登记。")?;
        let model = self.model_path(model_id)?.ok_or("打标模型尚未登记。")?;
        engines::get(engine_id)?.tag(self, &engine, &model, image, task)
    }
}
