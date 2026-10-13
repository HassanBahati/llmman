//! `llmman launch omp` (OMP coding agent).

use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::providers;

use super::common;
use super::{exec_with_env, find_on_path, has_flag, server, WINDOWS_PATH_EXTS};

/// omp: register llmman's endpoint and the selected model in `models.yml`.
///
/// OMP's built-in Ollama discovery sees llmman's `/api/tags`, but model
/// selection is resolved against its on-disk catalog before discovery has
/// populated it. A fresh HOME therefore rejects `--model ollama/<model>`.
/// Writing the provider and model explicitly, as Ollama's own launcher does,
/// makes the first launch work while preserving unrelated user configuration.
pub(super) fn launch_omp(
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_omp().ok_or_else(|| anyhow::anyhow!("omp is not installed"))?;
    let server = server();
    write_omp_config(model, reasons, vision, context_window, &server)?;
    exec_with_env(
        &bin,
        &omp_args(model, extra_args),
        &[("OLLAMA_BASE_URL", server.as_str())],
    )
}

const OMP_PROVIDER: &str = "ollama";
const OMP_SETUP_VERSION: u64 = 2;

/// `PATH`, then Bun's and the standalone installer's usual user-local bins.
pub(super) fn find_omp() -> Option<PathBuf> {
    find_on_path("omp").or_else(|| omp_fallback_paths().into_iter().find(|path| path.is_file()))
}

fn omp_fallback_paths() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return Vec::new();
    };
    let dirs = [
        home.join(".bun").join("bin"),
        home.join(".local").join("bin"),
    ];
    if cfg!(windows) {
        return dirs
            .into_iter()
            .flat_map(|dir| {
                WINDOWS_PATH_EXTS
                    .iter()
                    .map(move |ext| dir.join(format!("omp.{ext}")))
            })
            .collect();
    }
    dirs.into_iter().map(|dir| dir.join("omp")).collect()
}

/// OMP's agent directory. These are the paths OMP exposes for callers to
/// configure; profile layout remains OMP's responsibility.
pub(super) fn omp_agent_dir() -> anyhow::Result<PathBuf> {
    if let Some(dir) = common::configured_pi_agent_dir()? {
        return Ok(dir);
    }
    let home = crate::config::home_dir().context("no home directory")?;
    let config = std::env::var("PI_CONFIG_DIR")
        .ok()
        .filter(|dir| !dir.trim().is_empty())
        .unwrap_or_else(|| ".omp".to_string());
    let config = PathBuf::from(config.trim());
    Ok(if config.is_absolute() {
        config.join("agent")
    } else {
        home.join(config).join("agent")
    })
}

fn write_omp_config(
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
    server: &str,
) -> anyhow::Result<()> {
    let dir = omp_agent_dir()?;
    write_omp_config_in_dir(&dir, model, reasons, vision, context_window, server)
}

fn write_omp_config_in_dir(
    dir: &Path,
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
    server: &str,
) -> anyhow::Result<()> {
    write_omp_models_config_at(
        &dir.join("models.yml"),
        model,
        reasons,
        vision,
        context_window,
        server,
    )?;
    common::write_yaml_merged(&dir.join("config.yml"), "omp", omp_config_merged)
}

fn omp_config_merged(existing: &serde_json::Value) -> serde_json::Value {
    let mut config = existing.clone();
    if let Some(config) = config.as_object_mut() {
        let setup_version = config
            .get("setupVersion")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default()
            .max(OMP_SETUP_VERSION);
        config.insert("setupVersion".into(), serde_json::json!(setup_version));
    }
    config
}

fn write_omp_models_config_at(
    path: &Path,
    model: &str,
    reasons: bool,
    vision: bool,
    context_window: Option<u64>,
    server: &str,
) -> anyhow::Result<()> {
    let entry = common::pi_model_entry(model, reasons, vision, context_window);
    common::write_yaml_merged(path, "omp", |existing| {
        omp_models_merged(existing, server, &entry)
    })
}

/// Updates only llmman's `ollama` provider and the selected model. Other
/// providers, provider-specific options, and previously registered models are
/// retained; endpoint and protocol fields are owned by this launcher because
/// stale values would route the launch somewhere other than llmman.
fn omp_models_merged(
    existing: &serde_json::Value,
    server: &str,
    entry: &serde_json::Value,
) -> serde_json::Value {
    common::provider_models_merged(existing, OMP_PROVIDER, entry, true, |provider| {
        provider.insert(
            "baseUrl".into(),
            serde_json::json!(format!("{}/v1", server.trim_end_matches('/'))),
        );
        provider.insert("api".into(), serde_json::json!("openai-responses"));
        provider.remove("auth");
        provider.insert(
            "apiKey".into(),
            serde_json::json!(providers::PLACEHOLDER_API_KEY),
        );
        provider.insert("authHeader".into(), serde_json::json!(true));
        provider.insert("discovery".into(), serde_json::json!({ "type": "ollama" }));
    })
}

/// Select llmman's model through OMP's Ollama provider unless the caller
/// explicitly supplied an OMP model after `--`. Avoiding a duplicate matters
/// both for predictable precedence and for keeping OMP's argument parser out
/// of version-specific repeated-option behavior.
fn omp_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 2);
    if !has_flag(extra_args, "--model", Some("-m")) {
        args.extend(["--model".to_string(), format!("ollama/{model}")]);
    }
    args.extend_from_slice(extra_args);
    args
}

#[cfg(test)]
mod tests {
    use super::super::test_temp_dir;
    use super::super::{INTEGRATIONS, MODEL_FLAG_FORWARDED, MODEL_REQUIRED};
    use super::*;

    #[test]
    fn omp_is_listed_and_selects_the_model_through_ollama() {
        let omp = INTEGRATIONS.iter().find(|i| i.name == "omp").unwrap();
        assert_eq!(omp.binary, "omp");
        assert!(MODEL_REQUIRED.contains(&"omp"));
        assert!(MODEL_FLAG_FORWARDED.contains(&"omp"));

        assert_eq!(
            omp_args("docker.io/ai/qwen3.5:0.8b", &[]),
            vec!["--model", "ollama/docker.io/ai/qwen3.5:0.8b"]
        );

        let existing = serde_json::json!({
            "providers": {
                "other": { "baseUrl": "https://example.com" },
                "ollama": {
                    "auth": "none",
                    "headers": { "X-Custom": "kept" },
                    "models": [
                        {
                            "id": "docker.io/ai/qwen3.5:0.8b",
                            "name": "User name",
                            "input": ["text", "image"],
                            "maxTokens": 4096
                        },
                        { "id": "old", "name": "Old" }
                    ]
                }
            }
        });
        let entry = serde_json::json!({
            "id": "docker.io/ai/qwen3.5:0.8b",
            "name": "docker.io/ai/qwen3.5:0.8b",
            "input": ["text"]
        });
        let merged = omp_models_merged(&existing, "http://127.0.0.1:17434", &entry);
        let provider = &merged["providers"]["ollama"];
        assert_eq!(provider["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(provider["api"], "openai-responses");
        assert!(provider.get("auth").is_none());
        assert_eq!(provider["apiKey"], providers::PLACEHOLDER_API_KEY);
        assert_eq!(provider["authHeader"], true);
        assert_eq!(provider["discovery"]["type"], "ollama");
        assert_eq!(provider["headers"]["X-Custom"], "kept");
        assert_eq!(provider["models"][0]["name"], entry["name"]);
        assert_eq!(provider["models"][0]["input"], entry["input"]);
        assert_eq!(provider["models"][0]["maxTokens"], 4096);
        assert_eq!(provider["models"][1]["id"], "old");
        assert_eq!(
            merged["providers"]["other"]["baseUrl"],
            "https://example.com"
        );

        let config = omp_config_merged(&serde_json::json!({
            "setupVersion": OMP_SETUP_VERSION + 3,
            "theme": "dark"
        }));
        assert_eq!(config["setupVersion"], OMP_SETUP_VERSION + 3);
        assert_eq!(config["theme"], "dark");
    }

    #[test]
    fn omp_model_argument_after_separator_wins_without_a_duplicate() {
        let strings = |values: &[&str]| values.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        for forwarded in [
            strings(&["--model", "openrouter/anthropic/claude-sonnet-4"]),
            strings(&["-m", "ollama/other"]),
            strings(&["--model=ollama/other"]),
            strings(&["-m=ollama/other"]),
        ] {
            assert_eq!(omp_args("local", &forwarded), forwarded);
        }

        assert_eq!(
            omp_args("local", &strings(&["-p", "ping"])),
            strings(&["--model", "ollama/local", "-p", "ping"])
        );
    }

    #[test]
    fn omp_fallback_paths_cover_bun_and_local_bin() {
        let paths = omp_fallback_paths();
        let binary = if cfg!(windows) { "omp.exe" } else { "omp" };
        assert!(paths
            .iter()
            .any(|path| path.ends_with(Path::new(".bun").join("bin").join(binary))));
        assert!(paths
            .iter()
            .any(|path| path.ends_with(Path::new(".local").join("bin").join(binary))));
    }

    #[test]
    fn omp_config_is_created_for_a_fresh_home_and_round_trips() {
        let dir = test_temp_dir("omp-models");
        let models_path = dir.join("models.yml");
        write_omp_config_in_dir(
            &dir,
            "docker.io/ai/qwen3.5:0.8b",
            true,
            true,
            Some(32_768),
            "http://127.0.0.1:17434",
        )
        .unwrap();

        let parsed: serde_json::Value =
            yaml_serde::from_str(&std::fs::read_to_string(&models_path).unwrap()).unwrap();
        let provider = &parsed["providers"]["ollama"];
        assert_eq!(provider["baseUrl"], "http://127.0.0.1:17434/v1");
        assert_eq!(provider["models"][0]["id"], "docker.io/ai/qwen3.5:0.8b");
        assert_eq!(provider["models"][0]["name"], "docker.io/ai/qwen3.5:0.8b");
        assert_eq!(provider["models"][0]["reasoning"], true);
        assert_eq!(
            provider["models"][0]["input"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(provider["models"][0]["contextWindow"], 32_768);
        let config: serde_json::Value =
            yaml_serde::from_str(&std::fs::read_to_string(dir.join("config.yml")).unwrap())
                .unwrap();
        assert_eq!(config["setupVersion"], OMP_SETUP_VERSION);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn omp_models_yml_backs_up_the_exact_commented_file_before_rewriting() {
        let dir = test_temp_dir("omp-models-backup");
        let path = dir.join("models.yml");
        let config_path = dir.join("config.yml");
        let original = r#"# my own notes, do not delete
providers:
  openai:
    apiKey: sk-mine
defaults:
  temperature: 0.2   # tuned by hand
"#;
        std::fs::write(&path, original).unwrap();
        let config_original = "# keep this too\ntheme: dark\n";
        std::fs::write(&config_path, config_original).unwrap();

        write_omp_config_in_dir(
            &dir,
            "docker.io/ai/qwen3.5:0.8b",
            false,
            false,
            None,
            "http://127.0.0.1:17434",
        )
        .unwrap();
        write_omp_config_in_dir(
            &dir,
            "docker.io/ai/qwen3.5:0.8b",
            false,
            false,
            Some(4096),
            "http://127.0.0.1:17434",
        )
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(path.with_extension("yml.bak")).unwrap(),
            original
        );
        let mut hand_edited = std::fs::read_to_string(&path).unwrap();
        hand_edited.push_str("# added after the first launch\n");
        std::fs::write(&path, &hand_edited).unwrap();
        write_omp_config_in_dir(
            &dir,
            "docker.io/ai/qwen3.5:0.8b",
            false,
            false,
            Some(8192),
            "http://127.0.0.1:17434",
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(path.with_extension("yml.bak")).unwrap(),
            hand_edited
        );
        let parsed: serde_json::Value =
            yaml_serde::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(parsed["providers"]["openai"]["apiKey"], "sk-mine");
        assert_eq!(parsed["defaults"]["temperature"], 0.2);
        assert_eq!(
            parsed["providers"]["ollama"]["models"][0]["id"],
            "docker.io/ai/qwen3.5:0.8b"
        );
        assert_eq!(
            std::fs::read_to_string(config_path.with_extension("yml.bak")).unwrap(),
            config_original
        );
        let config: serde_json::Value =
            yaml_serde::from_str(&std::fs::read_to_string(config_path).unwrap()).unwrap();
        assert_eq!(config["theme"], "dark");
        assert_eq!(config["setupVersion"], OMP_SETUP_VERSION);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
