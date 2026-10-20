//! Memoized verification of installed weights.
//!
//! Verifying a model reads and hashes the whole file. The catalog is re-listed
//! on every refresh, and every readiness check before an inference call
//! resolves the model path again, so uncached verification hashes multi-gigabyte
//! weights repeatedly. Results are cached per model id, path, file size and
//! modification time: unchanged files are hashed once per process, and a file
//! whose size or modification time changed is verified again. Failures are
//! cached too, so a corrupt weight is not rehashed on every refresh.
//!
//! The cache is process-wide and keyed by absolute paths, so several
//! `ModelSdk` instances sharing weights also share results. Deleting a file
//! bypasses the cache because the fingerprint cannot be computed without it.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use crate::resources;

/// Upper bound before the cache is dropped wholesale. The built-in catalog is
/// small, so this only guards against repeatedly registering unrelated paths.
const MAX_ENTRIES: usize = 1024;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Stamp {
    name: String,
    size: u64,
    modified: Option<u128>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Key {
    id: String,
    path: PathBuf,
    stamps: Vec<Stamp>,
}

static CACHE: OnceLock<Mutex<HashMap<Key, Result<(), String>>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<Key, Result<(), String>>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Files that make up a model: the named files of a directory model, or `None`
/// for a single-file model, where the resolved path is the weight itself and
/// its file name is not part of the contract.
fn named_files(id: &str) -> Option<Vec<&'static str>> {
    if resources::wd14_spec(id).is_some() {
        Some(vec!["model.onnx", "selected_tags.csv"])
    } else if id == resources::OPENVINO_MODEL_ID {
        Some(resources::OPENVINO_MODEL_FILES.to_vec())
    } else if id == resources::KOKORO_MODEL_ID {
        Some(
            resources::KOKORO_FILES
                .iter()
                .map(|(name, _, _, _)| *name)
                .collect(),
        )
    } else {
        None
    }
}

fn stamp(name: &str, path: &Path) -> Option<Stamp> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    Some(Stamp {
        name: name.into(),
        size: metadata.len(),
        modified: metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|elapsed| elapsed.as_nanos()),
    })
}

/// Builds the cache key, or `None` when a file needed for the fingerprint is
/// missing; the caller then verifies without caching so the verifier reports
/// its own error.
fn key(id: &str, path: &Path) -> Option<Key> {
    let stamps = match named_files(id) {
        Some(names) => names
            .into_iter()
            .map(|name| stamp(name, &path.join(name)))
            .collect::<Option<Vec<_>>>()?,
        None => {
            let name = path.file_name()?.to_string_lossy().into_owned();
            vec![stamp(&name, path)?]
        }
    };
    Some(Key {
        id: id.into(),
        path: path.to_path_buf(),
        stamps,
    })
}

fn insert(key: Key, result: Result<(), String>) {
    let Ok(mut entries) = cache().lock() else {
        return;
    };
    if entries.len() >= MAX_ENTRIES && !entries.contains_key(&key) {
        entries.clear();
    }
    entries.insert(key, result);
}

/// Verifies a model, reusing a previous result for the same fingerprint.
pub(crate) fn cached(id: &str, path: &Path) -> Result<(), String> {
    let Some(key) = key(id, path) else {
        return verify(id, path);
    };
    match cache().lock() {
        Ok(entries) => {
            if let Some(result) = entries.get(&key) {
                return result.clone();
            }
        }
        Err(_) => return verify(id, path),
    }
    let result = verify(id, path);
    insert(key, result.clone());
    result
}

/// Verifies a model without consulting the cache.
pub(crate) fn verify(id: &str, path: &Path) -> Result<(), String> {
    #[cfg(test)]
    note_verification(path);
    if let Some(spec) = resources::wd14_spec(id) {
        resources::verify_wd14_model(path, spec)
    } else if id == resources::OPENVINO_MODEL_ID {
        resources::verify_openvino_model(path)?;
        for (name, hash) in resources::OPENVINO_MODEL_HASHES {
            crate::install::verify_sha256(&path.join(name), hash)?;
        }
        Ok(())
    } else if id == resources::KOKORO_MODEL_ID {
        resources::verify_kokoro_model(path)
    } else {
        resources::verify_model(path, resources::model_spec(id)?)
    }
}

#[cfg(test)]
static VERIFICATIONS: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();

#[cfg(test)]
fn note_verification(path: &Path) {
    let calls = VERIFICATIONS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut calls) = calls.lock() {
        *calls.entry(path.to_path_buf()).or_default() += 1;
    }
}

/// Counts how often a path was actually verified, keyed by path so tests do not
/// observe each other.
#[cfg(test)]
fn verifications(path: &Path) -> usize {
    VERIFICATIONS
        .get()
        .and_then(|calls| calls.lock().ok())
        .and_then(|calls| calls.get(path).copied())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest catalog model, so hashing a fake weight stays cheap.
    const MODEL_ID: &str = "yolo11n-det-fp32";
    const MODEL_SIZE: u64 = 10_756_872;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            Self(crate::test_support::scratch_path(&format!("verify-{name}")))
        }

        fn weight(&self) -> PathBuf {
            let path = self.0.join("renamed-weight.onnx");
            fs::write(&path, vec![0_u8; MODEL_SIZE as usize]).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn unchanged_weight_is_hashed_once() {
        let scratch = Scratch::new("hit");
        let weight = scratch.weight();
        assert!(cached(MODEL_ID, &weight)
            .unwrap_err()
            .contains("SHA-256 校验失败"));
        assert_eq!(verifications(&weight), 1);
        assert!(cached(MODEL_ID, &weight).is_err());
        assert!(cached(MODEL_ID, &weight).is_err());
        assert_eq!(verifications(&weight), 1, "重复校验没有复用缓存");
    }

    #[test]
    fn a_resized_weight_is_verified_again() {
        let scratch = Scratch::new("resize");
        let weight = scratch.weight();
        assert!(cached(MODEL_ID, &weight).is_err());
        let mut file = fs::OpenOptions::new().append(true).open(&weight).unwrap();
        std::io::Write::write_all(&mut file, &[0]).unwrap();
        drop(file);
        assert!(cached(MODEL_ID, &weight)
            .unwrap_err()
            .contains("大小不符"));
        assert_eq!(verifications(&weight), 2);
    }

    #[test]
    fn a_removed_weight_is_not_resurrected_by_the_cache() {
        let scratch = Scratch::new("removed");
        let weight = scratch.weight();
        assert!(cached(MODEL_ID, &weight)
            .unwrap_err()
            .contains("SHA-256 校验失败"));
        fs::remove_file(&weight).unwrap();
        let error = cached(MODEL_ID, &weight).unwrap_err();
        assert!(!error.contains("SHA-256"), "缓存返回了已删除文件的结果");
        assert_eq!(verifications(&weight), 2);
    }

    #[test]
    fn fingerprint_covers_size_and_modification_time() {
        let base = Stamp {
            name: "model.onnx".into(),
            size: 1,
            modified: Some(1),
        };
        assert_ne!(base, Stamp { size: 2, ..base.clone() });
        assert_ne!(base, Stamp { modified: Some(2), ..base.clone() });
    }

    #[test]
    fn directory_models_fingerprint_every_file() {
        let scratch = Scratch::new("pair");
        fs::write(scratch.0.join("model.onnx"), b"model").unwrap();
        assert!(key("wd-swinv2-tagger-v3", &scratch.0).is_none());
        fs::write(scratch.0.join("selected_tags.csv"), b"tags").unwrap();
        let first = key("wd-swinv2-tagger-v3", &scratch.0).unwrap();
        assert_eq!(first.stamps.len(), 2);
        fs::write(scratch.0.join("selected_tags.csv"), b"other").unwrap();
        let second = key("wd-swinv2-tagger-v3", &scratch.0).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn single_file_models_ignore_the_weight_file_name() {
        let scratch = Scratch::new("rename");
        let weight = scratch.weight();
        let key = key(MODEL_ID, &weight).unwrap();
        assert_eq!(key.stamps.len(), 1);
        assert_eq!(key.stamps[0].name, "renamed-weight.onnx");
        assert_eq!(key.stamps[0].size, MODEL_SIZE);
    }
}
