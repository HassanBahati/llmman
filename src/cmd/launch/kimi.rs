//! `llmman launch kimi` (Kimi Code CLI).

use super::{before_terminator, exec_with_env, find_on_path, has_flag, server};

/// The window used when llmman knows none: kimi's own with no model configured.
const KIMI_FALLBACK_CONTEXT_WINDOW: u64 = 100_000;

/// kimi's default `loop_control.reserved_context_size`.
const KIMI_RESERVED_CONTEXT_SIZE: u64 = 50_000;
/// The least kimi accepts for `loop_control.reserved_context_size`.
const KIMI_MIN_RESERVED_CONTEXT_SIZE: u64 = 1_000;

/// kimi: with a model, an inline `--config` serving only that model from
/// llmman's /v1. Without one, only OLLAMA_HOST, and kimi uses its own
/// configuration.
pub(super) fn launch_kimi(
    model: &str,
    api_key: &str,
    context_window: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_on_path("kimi").ok_or_else(|| anyhow::anyhow!("kimi is not installed"))?;
    let server = server();
    if model.is_empty() {
        return exec_with_env(&bin, extra_args, &[("OLLAMA_HOST", server.as_str())]);
    }
    let base_url = format!("{server}/v1");
    exec_with_env(
        &bin,
        &kimi_args(model, &base_url, context_window, extra_args),
        &kimi_env(&base_url, api_key),
    )
}

/// llmman's `--config`, then the caller's arguments.
fn kimi_args(
    model: &str,
    base_url: &str,
    context_window: Option<u64>,
    extra_args: &[String],
) -> Vec<String> {
    [
        vec![
            "--config".to_string(),
            kimi_config(model, base_url, context_window),
        ],
        extra_args.to_vec(),
    ]
    .concat()
}

/// kimi's OpenAI provider prefers these to the configuration's values, so
/// they are set to override inherited ones. This also keeps the key off
/// the command line.
fn kimi_env<'a>(base_url: &'a str, api_key: &'a str) -> [(&'static str, &'a str); 2] {
    [("OPENAI_BASE_URL", base_url), ("OPENAI_API_KEY", api_key)]
}

/// The `--config` JSON selecting `model` from an OpenAI-compatible
/// provider at `base_url`. Its key is empty: `OPENAI_API_KEY` supplies it.
fn kimi_config(model: &str, base_url: &str, context_window: Option<u64>) -> String {
    let window = context_window.unwrap_or(KIMI_FALLBACK_CONTEXT_WINDOW);
    serde_json::json!({
        "default_model": model,
        "models": {
            (model): {
                "provider": "llmman",
                "model": model,
                "max_context_size": window,
            },
        },
        "providers": {
            "llmman": {
                "type": "openai_legacy",
                "base_url": base_url,
                "api_key": "",
            },
        },
        "loop_control": { "reserved_context_size": reserved_context_size(window) },
    })
    .to_string()
}

/// A quarter of `window`, within kimi's bounds. kimi compacts when the
/// context plus this reserve fills the window, so its default would
/// compact a small window at every step.
fn reserved_context_size(window: u64) -> u64 {
    (window / 4).clamp(KIMI_MIN_RESERVED_CONTEXT_SIZE, KIMI_RESERVED_CONTEXT_SIZE)
}

/// Rejects kimi's own `--config`, `--config-file` and `--model`/`-m`
/// before a forwarded `--`, when llmman selects the model. `run` calls
/// this before the daemon starts.
pub(super) fn check_kimi_args(model: Option<&str>, extra_args: &[String]) -> anyhow::Result<()> {
    let Some(model) = model.map(str::trim).filter(|m| !m.is_empty()) else {
        return Ok(());
    };
    let args = before_terminator(extra_args);
    for flag in ["--config", "--config-file"] {
        anyhow::ensure!(
            !has_flag(args, flag, None),
            "kimi's {flag} would replace the configuration llmman passes for model {model}; \
             launch kimi without a model to use its own"
        );
    }
    anyhow::ensure!(
        !has_flag(args, "--model", Some("-m")),
        "kimi's --model names a model outside the configuration llmman passes; select it \
         with `llmman launch kimi --model <model>` instead"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn kimi_args_lead_with_llmmans_config() {
        let built = kimi_args("m", "http://h/v1", None, &args(&["--print", "-p", "hi"]));
        assert_eq!(built[0], "--config");
        assert_eq!(built[1], kimi_config("m", "http://h/v1", None));
        assert_eq!(built[2..], args(&["--print", "-p", "hi"]));
    }

    #[test]
    fn kimi_env_sets_the_base_url_and_key_kimi_prefers() {
        assert_eq!(
            kimi_env("http://h/v1", "k"),
            [("OPENAI_BASE_URL", "http://h/v1"), ("OPENAI_API_KEY", "k")]
        );
    }

    #[test]
    fn kimi_config_selects_the_model_at_llmmans_endpoint() {
        let config: serde_json::Value = serde_json::from_str(&kimi_config(
            "qwen3.5:0.8b",
            "http://127.0.0.1:11434/v1",
            Some(32_768),
        ))
        .unwrap();
        assert_eq!(config["default_model"], "qwen3.5:0.8b");
        let model = &config["models"]["qwen3.5:0.8b"];
        assert_eq!(model["provider"], "llmman");
        assert_eq!(model["model"], "qwen3.5:0.8b");
        assert_eq!(model["max_context_size"], 32_768);
        let provider = &config["providers"]["llmman"];
        assert_eq!(provider["type"], "openai_legacy");
        assert_eq!(provider["base_url"], "http://127.0.0.1:11434/v1");
        assert_eq!(provider["api_key"], "");
        assert_eq!(config["loop_control"]["reserved_context_size"], 8_192);
    }

    #[test]
    fn kimi_config_falls_back_to_kimis_window_when_there_is_none() {
        let config: serde_json::Value =
            serde_json::from_str(&kimi_config("m", "http://h/v1", None)).unwrap();
        assert_eq!(
            config["models"]["m"]["max_context_size"],
            KIMI_FALLBACK_CONTEXT_WINDOW
        );
    }

    #[test]
    fn reserved_context_size_stays_within_kimis_bounds() {
        assert_eq!(reserved_context_size(262_144), KIMI_RESERVED_CONTEXT_SIZE);
        assert_eq!(reserved_context_size(32_768), 8_192);
        assert_eq!(reserved_context_size(2_048), KIMI_MIN_RESERVED_CONTEXT_SIZE);
    }

    #[test]
    fn check_kimi_args_refuses_kimis_own_config_and_model_with_llmmans() {
        let err = |a: &[&str]| {
            check_kimi_args(Some("m"), &args(a))
                .unwrap_err()
                .to_string()
        };
        assert!(err(&["--config", "{}"]).contains("kimi's --config "));
        assert!(err(&["--config={}"]).contains("kimi's --config "));
        assert!(err(&["--config-file", "c.toml"]).contains("kimi's --config-file"));
        assert!(err(&["--model", "x"]).contains("kimi's --model"));
        assert!(err(&["-m", "x"]).contains("kimi's --model"));
        assert!(err(&["-m=x"]).contains("kimi's --model"));
    }

    #[test]
    fn check_kimi_args_leaves_kimis_own_selection_alone_without_a_model() {
        for model in [None, Some(""), Some("  ")] {
            assert!(check_kimi_args(model, &args(&["--config", "{}", "-m", "x"])).is_ok());
        }
    }

    #[test]
    fn check_kimi_args_stops_at_a_forwarded_terminator() {
        assert!(check_kimi_args(Some("m"), &args(&["--", "--model", "x"])).is_ok());
    }
}
