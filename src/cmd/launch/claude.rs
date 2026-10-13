//! `llmman launch claude`.

use super::{exec_with_env, find_on_path, has_forwarded_model_flag, server};

/// claude: set ANTHROPIC_BASE_URL and a dummy ANTHROPIC_API_KEY so it talks to
/// our server's Anthropic-compatible API.
pub(super) fn launch_claude(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("claude").ok_or_else(|| anyhow::anyhow!("claude is not installed"))?;
    let server = server();
    exec_with_env(
        &bin,
        &claude_args(model, extra_args),
        &claude_env(api_key, &server),
    )
}

/// Split out so a test can assert what claude is handed:
/// [`super::exec_with_env`] never returns, so calling [`launch_claude`] would
/// take the test runner with it.
fn claude_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    // Absent, not empty: claude reads "" as a model actually named "".
    // A forwarded `--model` replaces llmman's.
    if !model.is_empty() && !has_forwarded_model_flag("claude", extra_args) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// What llmman sets for claude. `base_url` is the bare origin: claude
/// appends `/v1/messages` itself. The model goes in [`claude_args`].
fn claude_env<'a>(api_key: &'a str, base_url: &'a str) -> Vec<(&'a str, &'a str)> {
    vec![
        ("ANTHROPIC_BASE_URL", base_url),
        ("ANTHROPIC_API_KEY", api_key),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_select_the_resolved_model_and_pass_the_rest_through() {
        assert_eq!(
            claude_args("local/model", &["-p".into(), "hello".into()]),
            ["--model", "local/model", "-p", "hello"]
        );
        assert_eq!(
            claude_args("", &["-p".into(), "hello".into()]),
            ["-p", "hello"]
        );
        // `run` prepends `effort_args`, so `--effort` arrives in `extra_args`.
        assert_eq!(
            claude_args(
                "local/model",
                &["--effort".into(), "high".into(), "-p".into(), "x".into()]
            ),
            ["--model", "local/model", "--effort", "high", "-p", "x"]
        );
        // A forwarded `--model` replaces llmman's.
        assert_eq!(
            claude_args("local/model", &["--model".into(), "other/model".into()]),
            ["--model", "other/model"]
        );
        // claude has no `-m`, so it is not read as a model.
        assert_eq!(
            claude_args("local/model", &["-m".into(), "not-a-model-flag".into()]),
            ["--model", "local/model", "-m", "not-a-model-flag"]
        );
    }

    /// A wrong or missing base URL sends the session to api.anthropic.com
    /// instead of the daemon.
    #[test]
    fn env_points_at_the_daemon_and_carries_its_key() {
        let env = claude_env("secret", "http://127.0.0.1:17434");
        let get = |key| env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
        assert_eq!(get("ANTHROPIC_BASE_URL"), Some("http://127.0.0.1:17434"));
        assert_eq!(get("ANTHROPIC_API_KEY"), Some("secret"));
        assert_eq!(env.len(), 2, "llmman sets no other variables");
    }
}
