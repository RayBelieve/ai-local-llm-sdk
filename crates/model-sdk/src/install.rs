mod crispasr;
mod ffmpeg;
mod llama;
mod onnxruntime;
mod openvino;

use std::{
    fs::{self, File},
    future::Future,
    io::{Read, Write},
    path::Path,
    time::Duration,
};

use sha2::{Digest, Sha256};

use crate::{network, task::TaskControl};

pub(crate) fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    if format!("{:x}", digest.finalize()) != expected {
        return Err(format!("资源校验失败: {}", path.display()));
    }
    Ok(())
}

fn replace_file(staged: &Path, target: &Path) -> Result<(), String> {
    let backup = target.with_extension("backup");
    if target.exists() {
        if backup.exists() {
            fs::remove_file(&backup).map_err(|error| error.to_string())?;
        }
        fs::rename(target, &backup).map_err(|error| error.to_string())?;
    }
    if let Err(error) = fs::rename(staged, target) {
        if backup.exists() {
            let _ = fs::rename(&backup, target);
        }
        return Err(error.to_string());
    }
    if backup.exists() {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

fn download_to(url: &str, target: &Path, stage: &str, task: &TaskControl) -> Result<(), String> {
    download_to_range(url, target, stage, 0, 100, task)
}

async fn await_or_cancel<T>(
    task: &TaskControl,
    operation: impl Future<Output = T>,
) -> Result<T, String> {
    task.check_cancelled()?;
    tokio::pin!(operation);
    tokio::select! {
        biased;
        _ = task.cancelled() => Err(crate::task::CANCELLED_ERROR.into()),
        result = &mut operation => Ok(result),
    }
}

fn block_on<T: Send>(future: impl Future<Output = T> + Send) -> Result<T, String> {
    if tokio::runtime::Handle::try_current().is_ok() {
        std::thread::scope(|scope| {
            scope
                .spawn(|| block_on_runtime(future))
                .join()
                .map_err(|_| "下载线程异常退出。".to_owned())?
        })
    } else {
        block_on_runtime(future)
    }
}

fn block_on_runtime<T>(future: impl Future<Output = T>) -> Result<T, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    let result = runtime.block_on(future);
    runtime.shutdown_timeout(Duration::from_secs(1));
    Ok(result)
}

fn download_to_range(
    url: &str,
    target: &Path,
    stage: &str,
    from: u8,
    to: u8,
    task: &TaskControl,
) -> Result<(), String> {
    task.check_cancelled()?;
    let part = target.with_extension("part");
    let result = block_on(async {
        let mut response = await_or_cancel(task, network::download_client()?.get(url).send())
            .await?
            .map_err(|error| format!("下载连接失败: {error}"))?
            .error_for_status()
            .map_err(|error| format!("下载失败: {error}"))?;
        let expected = response.content_length().unwrap_or(0);
        let mut file = File::create(&part).map_err(|error| error.to_string())?;
        let mut received = 0_u64;
        let mut previous = 255;
        loop {
            let Some(chunk) = await_or_cancel(task, response.chunk())
                .await?
                .map_err(|error| error.to_string())?
            else {
                break;
            };
            file.write_all(&chunk).map_err(|error| error.to_string())?;
            received += chunk.len() as u64;
            let percent = if expected == 0 {
                from
            } else {
                from + ((received * u64::from(to - from) / expected).min(u64::from(to - from))
                    as u8)
            };
            if percent != previous {
                task.progress(
                    stage,
                    percent,
                    &format!("已下载 {:.1} MB", received as f64 / 1_048_576.0),
                );
                previous = percent;
            }
        }
        file.sync_all().map_err(|error| error.to_string())?;
        if expected != 0 && received != expected {
            return Err(format!("下载不完整: 收到 {received} / {expected} 字节"));
        }
        Ok::<(), String>(())
    })?;
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result?;
    if let Err(error) = task.check_cancelled() {
        let _ = fs::remove_file(&part);
        return Err(error);
    }
    replace_file(&part, target)
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        path::PathBuf,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        },
        thread,
        time::Duration,
    };

    use crate::task::{TaskControl, CANCELLED_ERROR};

    static SCRATCH_IDS: AtomicU64 = AtomicU64::new(0);

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let id = SCRATCH_IDS.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("model-sdk-download-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn read_headers(socket: &mut std::net::TcpStream) {
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut buffer = Vec::new();
        let mut byte = [0_u8; 1];
        while !buffer.ends_with(b"\r\n\r\n") {
            match socket.read(&mut byte) {
                Ok(1) => buffer.push(byte[0]),
                _ => break,
            }
            if buffer.len() > 16 * 1024 {
                break;
            }
        }
    }

    fn serve(body: Vec<u8>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            read_headers(&mut socket);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(header.as_bytes()).unwrap();
            socket.write_all(&body).unwrap();
        });
        port
    }

    #[test]
    fn downloads_the_response_body() {
        let body = b"hello-model".to_vec();
        let port = serve(body.clone());
        let scratch = Scratch::new();
        let target = scratch.0.join("model.bin");
        super::download_to(
            &format!("http://127.0.0.1:{port}/model.bin"),
            &target,
            "model",
            &TaskControl::default(),
        )
        .unwrap();
        assert_eq!(std::fs::read(target).unwrap(), body);
    }

    #[test]
    fn download_works_on_a_thread_that_already_runs_tokio() {
        let body = b"runtime-body".to_vec();
        let port = serve(body.clone());
        let scratch = Scratch::new();
        let target = scratch.0.join("model.bin");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime
            .block_on(async {
                super::download_to(
                    &format!("http://127.0.0.1:{port}/model.bin"),
                    &target,
                    "model",
                    &TaskControl::default(),
                )
            })
            .unwrap();
        assert_eq!(std::fs::read(scratch.0.join("model.bin")).unwrap(), body);
    }

    #[test]
    fn incomplete_response_removes_the_partial_file() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            read_headers(&mut socket);
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nhi")
                .unwrap();
        });
        let scratch = Scratch::new();
        let target = scratch.0.join("model.bin");
        let error = super::download_to(
            &format!("http://127.0.0.1:{port}/model.bin"),
            &target,
            "model",
            &TaskControl::default(),
        )
        .unwrap_err();
        assert!(!error.is_empty());
        assert!(!target.exists());
        assert!(!scratch.0.join("model.part").exists());
    }

    #[test]
    fn cancelled_task_fails_before_connecting() {
        let task = TaskControl::default();
        task.cancel().unwrap();
        let scratch = Scratch::new();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = done_tx.send(super::download_to(
                "http://127.0.0.1:1/file.bin",
                &scratch.0.join("file.bin"),
                "model",
                &task,
            ));
        });
        let error = done_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("cancelled download did not return")
            .unwrap_err();
        assert_eq!(error, CANCELLED_ERROR);
    }

    #[test]
    fn cancellation_aborts_an_open_download() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            read_headers(&mut socket);
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 1048576\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
            socket.write_all(&[1_u8; 4096]).unwrap();
            let _ = ready_tx.send(());
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut buffer = [0_u8; 8];
            let _ = socket.read(&mut buffer);
        });
        let task = Arc::new(TaskControl::default());
        let running = task.clone();
        let scratch = Scratch::new();
        let target = scratch.0.join("model.bin");
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = done_tx.send(super::download_to(
                &format!("http://127.0.0.1:{port}/model.bin"),
                &target,
                "model",
                &running,
            ));
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        task.cancel().unwrap();
        let error = done_rx
            .recv_timeout(Duration::from_secs(3))
            .expect("cancel did not abort the download")
            .unwrap_err();
        assert_eq!(error, CANCELLED_ERROR);
        assert!(!scratch.0.join("model.bin").exists());
        assert!(!scratch.0.join("model.part").exists());
        server.join().unwrap();
    }
}
