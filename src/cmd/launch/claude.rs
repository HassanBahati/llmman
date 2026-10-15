//! `llmman launch claude`.

use super::common;
use super::{
    before_terminator, exec_with_env_removing, find_on_path, has_forwarded_model_flag, server,
};

/// Inherited variables that point claude at another provider or credential.
const CLAUDE_ENV_TO_CLEAR: &[&str] = &[
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_GATEWAY",
];

/// claude's alias, background and subagent models. ANTHROPIC_SMALL_FAST_MODEL
/// is read before ANTHROPIC_DEFAULT_HAIKU_MODEL.
const CLAUDE_MODEL_ENV: &[&str] = &[
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
];

/// claude's model aliases, each with an optional `[1m]` suffix, and
/// `default`. They resolve through [`CLAUDE_MODEL_ENV`].
const CLAUDE_MODEL_ALIASES: &[&str] = &[
    "default", "best", "fable", "opus", "sonnet", "haiku", "opusplan",
];

/// claude: set ANTHROPIC_BASE_URL and a dummy ANTHROPIC_API_KEY so it talks to
/// our server's Anthropic-compatible API.
pub(super) fn launch_claude(
    model: &str,
    api_key: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("claude").ok_or_else(|| anyhow::anyhow!("claude is not installed"))?;
    let server = server();
    exec_with_env_removing(
        &bin,
        &claude_args(model, extra_args),
        &claude_env(session_model(model, extra_args), api_key, &server),
        CLAUDE_ENV_TO_CLEAR,
    )
}

/// Split out so a test can assert what claude is handed:
/// [`super::exec_with_env_removing`] never returns, so calling
/// [`launch_claude`] would take the test runner with it.
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

/// The model to set [`CLAUDE_MODEL_ENV`] to: a forwarded `--model`, unless
/// it is an alias, else `model`. Empty when a forwarded `--model` has no value.
fn session_model<'a>(model: &'a str, extra_args: &'a [String]) -> &'a str {
    if !has_forwarded_model_flag("claude", extra_args) {
        return model;
    }
    match common::forwarded_model(before_terminator(extra_args)) {
        Some(forwarded) if !is_claude_alias(forwarded) => forwarded,
        Some(_) => model,
        None => "",
    }
}

fn is_claude_alias(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let name = name.strip_suffix("[1m]").unwrap_or(&name);
    CLAUDE_MODEL_ALIASES.contains(&name)
}

/// What llmman sets for claude. `base_url` is the bare origin: claude
/// appends `/v1/messages` itself. Every [`CLAUDE_MODEL_ENV`] is set to
/// `model`, so background requests and subagents use it too.
fn claude_env<'a>(model: &'a str, api_key: &'a str, base_url: &'a str) -> Vec<(&'a str, &'a str)> {
    let mut env = vec![
        ("ANTHROPIC_BASE_URL", base_url),
        ("ANTHROPIC_API_KEY", api_key),
    ];
    if !model.is_empty() {
        env.extend(CLAUDE_MODEL_ENV.iter().map(|&name| (name, model)));
    }
    env
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

    #[test]
    fn session_model_is_the_forwarded_one_when_there_is_one() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            session_model("local/model", &args(&["-p", "x"])),
            "local/model"
        );
        assert_eq!(
            session_model("local/model", &args(&["--model", "other/model"])),
            "other/model"
        );
        assert_eq!(
            session_model("local/model", &args(&["--model=other/model"])),
            "other/model"
        );
        assert_eq!(
            session_model("", &args(&["--model", "other/model"])),
            "other/model"
        );
        assert_eq!(session_model("local/model", &args(&["--model="])), "");
        // An alias resolves through the variables, so they keep `model`.
        for alias in ["sonnet", "Opus", "fable[1m]", "opusplan", "default"] {
            assert_eq!(
                session_model("local/model", &args(&["--model", alias])),
                "local/model",
                "{alias}"
            );
        }
        // With no `model`, an alias leaves the variables unset.
        assert_eq!(session_model("", &args(&["--model", "default"])), "");
        assert_eq!(
            session_model("local/model", &args(&["--model", "sonnet-local"])),
            "sonnet-local"
        );
        // After a second `--` it is a message, not a flag.
        assert_eq!(
            session_model(
                "local/model",
                &args(&["-p", "--", "--model", "other/model"])
            ),
            "local/model"
        );
    }

    /// A wrong or missing base URL sends the session to api.anthropic.com
    /// instead of the daemon.
    #[test]
    fn env_points_at_the_daemon_and_carries_its_key() {
        let env = claude_env("", "secret", "http://127.0.0.1:17434");
        let get = |key| env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
        assert_eq!(get("ANTHROPIC_BASE_URL"), Some("http://127.0.0.1:17434"));
        assert_eq!(get("ANTHROPIC_API_KEY"), Some("secret"));
        assert_eq!(env.len(), 2, "without a model, llmman sets nothing else");
    }

    #[test]
    fn env_pins_every_alias_and_subagents_to_the_resolved_model() {
        let env = claude_env("local/model", "secret", "http://127.0.0.1:17434");
        for key in [
            "ANTHROPIC_DEFAULT_FABLE_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ] {
            let value = env.iter().find(|(name, _)| *name == key).map(|(_, v)| *v);
            assert_eq!(value, Some("local/model"), "{key}");
        }
        assert_eq!(env.len(), 2 + CLAUDE_MODEL_ENV.len());
    }

    /// Removal runs last, so an overlap would unset llmman's value.
    #[test]
    fn cleared_env_overlaps_nothing_llmman_sets() {
        let env = claude_env("local/model", "secret", "http://127.0.0.1:17434");
        for key in CLAUDE_ENV_TO_CLEAR {
            assert!(env.iter().all(|(name, _)| name != key), "{key}");
        }
    }
}
