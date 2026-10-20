//! Scratch directories for unit tests.
//!
//! `std::env::temp_dir()` is redirected or simply not writable for restricted
//! processes (sandboxed runners, service accounts), which makes tests that only
//! need a directory fail for reasons unrelated to the code under test. Tests
//! therefore prefer a directory inside the crate's target directory and fall
//! back to the platform temp directory only when that cannot be created.

use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Creates a unique directory for one test and returns its path. Callers keep
/// ownership of the cleanup, so tests that hand the path to an SDK instance can
/// remove it themselves once the instance is dropped.
pub(crate) fn scratch_path(name: &str) -> PathBuf {
    let path = scratch_root().join(format!(
        "{name}-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).expect("无法创建测试临时目录");
    path
}

fn scratch_root() -> PathBuf {
    [
        std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target")),
        Some(std::env::temp_dir()),
    ]
    .into_iter()
    .flatten()
    .map(|root| root.join("model-sdk-scratch"))
    .find(|root| fs::create_dir_all(root).is_ok())
    .expect("没有可写的测试临时目录")
}
