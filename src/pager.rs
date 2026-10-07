use std::io::{self, IsTerminal, Write};
use std::process::{Command, Stdio};

/// Through the pager when stdout is a terminal, as git: `$LLMMAN_PAGER`,
/// then the command-specific pager, `core.pager`, `$PAGER`, or `less`.
/// Empty or `cat` means none; false boolean values in `pager.<command>`
/// disable paging even when `$LLMMAN_PAGER` is set.
/// `LESS=FRX` is git's default too.
pub(crate) fn emit(subcommand: &str, text: &str, pager: bool) -> anyhow::Result<()> {
    if pager && io::stdout().is_terminal() {
        if let Some(mut child) = pager_command(subcommand).and_then(|cmd| spawn_pager(&cmd).ok()) {
            if let Some(mut stdin) = child.stdin.take() {
                // Quitting the pager early closes the pipe; not an error.
                let _ = stdin.write_all(text.as_bytes());
            }
            // 127 (sh) / 9009 (cmd): no such pager, nothing was shown.
            if !matches!(child.wait()?.code(), Some(127 | 9009)) {
                return Ok(());
            }
        }
    }
    match io::stdout().write_all(text.as_bytes()) {
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

fn pager_command(subcommand: &str) -> Option<String> {
    let (core, specific) = crate::config::pager(subcommand);
    resolve(
        std::env::var("LLMMAN_PAGER").ok().as_deref(),
        core.as_deref(),
        specific.as_deref(),
        std::env::var("PAGER").ok().as_deref(),
    )
}

fn resolve(
    env: Option<&str>,
    core: Option<&str>,
    specific: Option<&str>,
    pager: Option<&str>,
) -> Option<String> {
    let enabled = specific.and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Some(true),
        "false" | "no" | "off" | "0" => Some(false),
        _ => None,
    });
    if enabled == Some(false) {
        return None;
    }
    let specific = specific.filter(|_| enabled.is_none());
    let cmd = env.or(specific).or(core).or(pager).unwrap_or("less").trim();
    (!cmd.is_empty() && cmd != "cat").then(|| cmd.to_string())
}

fn spawn_pager(cmd: &str) -> io::Result<std::process::Child> {
    let mut command = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", cmd]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", cmd]);
        c
    };
    for (var, default) in [("LESS", "FRX"), ("LV", "-c")] {
        if std::env::var_os(var).is_none() {
            command.env(var, default);
        }
    }
    command.stdin(Stdio::piped()).spawn()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pager_precedence_and_opt_out_match_git() {
        assert_eq!(resolve(None, None, None, None).as_deref(), Some("less"));
        assert_eq!(
            resolve(None, None, None, Some("more")).as_deref(),
            Some("more")
        );
        assert_eq!(
            resolve(None, Some("less -S"), None, Some("more")).as_deref(),
            Some("less -S")
        );
        assert_eq!(
            resolve(None, Some("less"), Some("more"), None).as_deref(),
            Some("more")
        );
        assert_eq!(
            resolve(Some("env-pager"), Some("less"), Some("more"), None).as_deref(),
            Some("env-pager")
        );
        assert_eq!(
            resolve(None, Some("less"), Some("true"), None).as_deref(),
            Some("less")
        );
        for disabled in ["", "cat", "  cat  "] {
            assert_eq!(resolve(Some(disabled), Some("less"), None, None), None);
        }
        for value in ["false", "no", "off", "0", "FALSE", " No ", "OFF"] {
            assert_eq!(resolve(Some("less"), None, Some(value), None), None);
        }
        for value in ["true", "yes", "on", "1", "TRUE", " Yes ", "ON"] {
            assert_eq!(
                resolve(None, Some("less -S"), Some(value), None).as_deref(),
                Some("less -S")
            );
        }
        assert_eq!(resolve(None, Some("less"), Some(""), None), None);
    }
}
