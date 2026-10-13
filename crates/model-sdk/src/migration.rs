use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::resources::{load_selection, save_selection};

pub fn legacy_resources_available(dir: &Path) -> bool {
    [dir.join("models"), dir.join("engines/crispasr")]
        .iter()
        .any(|path| fs::read_dir(path).is_ok_and(|mut entries| entries.next().is_some()))
}

fn file_hash(path: &Path) -> Result<Vec<u8>, String> {
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
    Ok(hash.finalize().to_vec())
}

fn same_file(source: &Path, target: &Path) -> Result<bool, String> {
    let source_size = fs::metadata(source)
        .map_err(|error| error.to_string())?
        .len();
    let target_size = fs::metadata(target)
        .map_err(|error| error.to_string())?
        .len();
    Ok(source_size == target_size && file_hash(source)? == file_hash(target)?)
}

fn move_managed_file(source: &Path, target: &Path) -> Result<(), String> {
    if target.exists() {
        if !same_file(source, target)? {
            return Err(format!(
                "资源迁移遇到同名但内容不同的文件: {}",
                target.display()
            ));
        }
    } else {
        fs::create_dir_all(target.parent().ok_or("资源目标路径无效。")?)
            .map_err(|error| error.to_string())?;
        if fs::rename(source, target).is_ok() {
            return Ok(());
        }
        let mut temporary_name = target.as_os_str().to_os_string();
        temporary_name.push(".migrating");
        let temporary = PathBuf::from(temporary_name);
        fs::copy(source, &temporary).map_err(|error| error.to_string())?;
        if !same_file(source, &temporary)? {
            let _ = fs::remove_file(&temporary);
            return Err(format!("资源复制校验失败: {}", source.display()));
        }
        fs::rename(&temporary, target).map_err(|error| error.to_string())?;
    }
    fs::remove_file(source).map_err(|error| error.to_string())
}

fn move_managed_tree(source: &Path, target: &Path) -> Result<(), String> {
    if !source.exists() {
        return Ok(());
    }
    fs::create_dir_all(target).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            move_managed_tree(&entry.path(), &destination)?;
        } else if kind.is_file() {
            move_managed_file(&entry.path(), &destination)?;
        } else {
            return Err(format!("无法迁移特殊文件: {}", entry.path().display()));
        }
    }
    fs::remove_dir(source).map_err(|error| error.to_string())
}

pub fn migrate_legacy_resources(legacy: &Path, portable: &Path) -> Result<(), String> {
    if legacy == portable || !legacy_resources_available(legacy) {
        return Ok(());
    }
    if !portable.join("resources.json").exists() {
        let mut selection = load_selection(legacy)?;
        for path in selection
            .engine_paths
            .values_mut()
            .chain(selection.model_paths.values_mut())
        {
            if let Ok(relative) = Path::new(path).strip_prefix(legacy) {
                *path = portable.join(relative).to_string_lossy().into_owned();
            }
        }
        save_selection(portable, &selection)?;
    }
    move_managed_tree(&legacy.join("models"), &portable.join("models"))?;
    move_managed_tree(
        &legacy.join("engines/crispasr"),
        &portable.join("engines/crispasr"),
    )?;
    let legacy_engine_dir = legacy.join("engines");
    if legacy_engine_dir.is_dir()
        && fs::read_dir(&legacy_engine_dir).is_ok_and(|mut entries| entries.next().is_none())
    {
        let _ = fs::remove_dir(legacy_engine_dir);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::resources::{engine_path, Selection};
    #[test]
    fn migrates_legacy_resources_and_selection() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subtitle-migration-test-{nonce}"));
        let legacy = root.join("legacy");
        let portable = root.join("portable");
        fs::create_dir_all(legacy.join("engines/crispasr")).unwrap();
        fs::create_dir_all(legacy.join("models")).unwrap();
        fs::write(legacy.join("engines/crispasr/backend.txt"), "vulkan").unwrap();
        fs::write(legacy.join("engines/crispasr/crispasr.exe"), b"engine").unwrap();
        fs::write(legacy.join("models/ggml-base.bin"), b"model").unwrap();
        save_selection(&legacy, &Selection::default()).unwrap();

        migrate_legacy_resources(&legacy, &portable).unwrap();

        assert!(!legacy_resources_available(&legacy));
        assert_eq!(
            fs::read(portable.join("models/ggml-base.bin")).unwrap(),
            b"model"
        );
        let selection = load_selection(&portable).unwrap();
        assert_eq!(selection.engine_id, "vulkan");
        assert_eq!(selection.model_id, "base");
        assert!(engine_path(&portable, &selection, "vulkan").is_some());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn migration_keeps_source_on_conflict() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subtitle-migration-conflict-{nonce}"));
        let legacy = root.join("legacy");
        let portable = root.join("portable");
        fs::create_dir_all(legacy.join("models")).unwrap();
        fs::create_dir_all(portable.join("models")).unwrap();
        fs::write(legacy.join("models/ggml-base.bin"), b"old").unwrap();
        fs::write(portable.join("models/ggml-base.bin"), b"new").unwrap();

        assert!(migrate_legacy_resources(&legacy, &portable).is_err());
        assert_eq!(
            fs::read(legacy.join("models/ggml-base.bin")).unwrap(),
            b"old"
        );
        assert_eq!(
            fs::read(portable.join("models/ggml-base.bin")).unwrap(),
            b"new"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
