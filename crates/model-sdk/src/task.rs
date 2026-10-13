use std::{
    process::{Child, Command, ExitStatus},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    thread,
    time::Duration,
};

pub struct TaskControl {
    child: Mutex<Option<Child>>,
    cancelled: AtomicBool,
    progress: Box<dyn Fn(&str, u8, &str) + Send + Sync>,
}

impl TaskControl {
    pub fn new(progress: impl Fn(&str, u8, &str) + Send + Sync + 'static) -> Self {
        Self {
            child: Mutex::new(None),
            cancelled: AtomicBool::new(false),
            progress: Box::new(progress),
        }
    }

    pub fn progress(&self, stage: &str, percent: u8, message: &str) {
        (self.progress)(stage, percent, message);
    }

    pub fn check_cancelled(&self) -> Result<(), String> {
        if self.cancelled.load(Ordering::SeqCst) {
            Err("任务已取消。".into())
        } else {
            Ok(())
        }
    }

    pub fn run(&self, mut command: Command) -> Result<ExitStatus, String> {
        self.check_cancelled()?;
        {
            let mut active = self.child.lock().map_err(|error| error.to_string())?;
            self.check_cancelled()?;
            if active.is_some() {
                return Err("已有任务正在运行。".into());
            }
            *active = Some(
                command
                    .spawn()
                    .map_err(|error| format!("启动进程失败: {error}"))?,
            );
        }
        loop {
            thread::sleep(Duration::from_millis(250));
            let mut active = self.child.lock().map_err(|error| error.to_string())?;
            match active.as_mut() {
                Some(child) => match child.try_wait() {
                    Ok(Some(status)) => {
                        *active = None;
                        return Ok(status);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        *active = None;
                        return Err(error.to_string());
                    }
                },
                None => return Err("任务已取消。".into()),
            }
        }
    }

    pub fn cancel(&self) -> Result<(), String> {
        self.cancelled.store(true, Ordering::SeqCst);
        let mut active = self.child.lock().map_err(|error| error.to_string())?;
        if let Some(mut child) = active.take() {
            child.kill().map_err(|error| error.to_string())?;
            let _ = child.wait();
        }
        Ok(())
    }
}

impl Default for TaskControl {
    fn default() -> Self {
        Self::new(|_, _, _| {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_stops_a_pending_task() {
        let task = TaskControl::default();
        task.cancel().unwrap();
        assert!(task.check_cancelled().is_err());
    }

    #[test]
    fn cancellation_stops_a_child_process() {
        let task = std::sync::Arc::new(TaskControl::default());
        let running = task.clone();
        let child = std::thread::spawn(move || {
            #[cfg(windows)]
            let command = {
                let mut command = Command::new("powershell");
                command.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 5"]);
                command
            };
            #[cfg(not(windows))]
            let command = {
                let mut command = Command::new("sleep");
                command.arg("5");
                command
            };
            running.run(command)
        });
        std::thread::sleep(Duration::from_millis(100));
        task.cancel().unwrap();
        assert!(child.join().unwrap().is_err());
    }
}
