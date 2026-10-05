//! `llmman launch grok` (Grok Build).

use std::path::{Path, PathBuf};

use anyhow::Context;

use super::common;
use super::{env_dir, exec_with_env, find_on_path, has_flag, server, Effort};

/// The per-model `env_key` in llmman's Grok config reads this variable.
/// A model credential outranks both Grok's signed-in session and its global
/// `XAI_API_KEY`, without putting the actual key on disk.
const GROK_API_KEY_ENV: &str = "LLMMAN_GROK_API_KEY";

/// grok: point its custom-model catalog and inference client at llmman's
/// OpenAI-compatible surface. Every auxiliary model is pinned too: without
/// this, Grok Build keeps built-in hosted ids for title/summary, image
/// description, web search, and prompt suggestions, then asks the local
/// daemon to load one after the main model already answered successfully.
///
/// The model flag is injected only when the caller did not provide one
/// after `--`. This matches qwen's behavior in `super` and lets an explicit
/// integration argument win without passing a duplicate flag.
pub(super) fn launch_grok(
    model: &str,
    api_key: &str,
    effort: Option<&Effort>,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let bin = find_grok().ok_or_else(|| anyhow::anyhow!("grok is not installed"))?;
    let effective_model = common::forwarded_model(extra_args).unwrap_or(model);
    let base_url = format!("{}/v1", server());
    let models_url = format!("{base_url}/models");
    // Never edit the user's config.toml. This child is wholly llmman-owned,
    // and setting GROK_HOME below scopes it to this launched process.
    let home = grok_home()?.join("llmman");
    let efforts = effort.map_or(&[][..], |e| &e.levels);
    write_grok_config(&home, effective_model, &base_url, efforts)?;
    let home = home.to_string_lossy().into_owned();
    let args = grok_args(model, extra_args);
    exec_with_env(
        &bin,
        &args,
        &grok_env(effective_model, api_key, &base_url, &models_url, &home),
    )
}

fn grok_env<'a>(
    model: &'a str,
    api_key: &'a str,
    base_url: &'a str,
    models_url: &'a str,
    home: &'a str,
) -> Vec<(&'a str, &'a str)> {
    vec![
        ("GROK_HOME", home),
        ("GROK_MODELS_BASE_URL", base_url),
        // Override an inherited custom catalog too. If it points elsewhere,
        // the selected local model is absent and Grok refuses `--model`
        // before making an inference request.
        ("GROK_MODELS_LIST_URL", models_url),
        ("GROK_DEFAULT_MODEL", model),
        ("GROK_WEB_SEARCH_MODEL", model),
        ("GROK_SESSION_SUMMARY_MODEL", model),
        ("GROK_IMAGE_DESCRIPTION_MODEL", model),
        ("GROK_PROMPT_SUGGESTIONS_MODEL", model),
        (GROK_API_KEY_ENV, api_key),
        // Grok uses this global fallback while fetching the remote catalog;
        // inference uses the higher-priority per-model env_key above.
        ("XAI_API_KEY", api_key),
    ]
}

/// Grok's configured home, or its documented `~/.grok` default.
pub(super) fn grok_home() -> anyhow::Result<PathBuf> {
    if let Some(path) = env_dir("GROK_HOME") {
        return Ok(path);
    }
    Ok(dirs::home_dir().context("no home directory")?.join(".grok"))
}

fn write_grok_config(
    home: &Path,
    model: &str,
    base_url: &str,
    efforts: &[&str],
) -> anyhow::Result<()> {
    let path = home.join("config.toml");
    std::fs::create_dir_all(home).with_context(|| format!("create {}", home.display()))?;
    let contents = grok_config_document(model, base_url, efforts);
    crate::fsutil::write_atomic(&path, contents.as_bytes())
        .with_context(|| format!("write {}", path.display()))
}

/// Grok Build ignores `--effort` for a model without `efforts` listed.
fn grok_config_document(model: &str, base_url: &str, efforts: &[&str]) -> String {
    let mut entry = toml_edit::Table::new();
    entry["model"] = toml_edit::value(model);
    entry["base_url"] = toml_edit::value(base_url);
    entry["env_key"] = toml_edit::value(GROK_API_KEY_ENV);
    entry["api_backend"] = toml_edit::value("chat_completions");
    if !efforts.is_empty() {
        entry["reasoning_efforts"] =
            toml_edit::value(efforts.iter().copied().collect::<toml_edit::Array>());
    }

    let mut models = toml_edit::Table::new();
    models.insert(model, toml_edit::Item::Table(entry));
    let mut document = toml_edit::DocumentMut::new();
    document.insert("model", toml_edit::Item::Table(models));
    document.to_string()
}

fn grok_args(model: &str, extra_args: &[String]) -> Vec<String> {
    let mut args = Vec::with_capacity(extra_args.len() + 2);
    if !has_flag(extra_args, "--model", Some("-m")) {
        args.extend(["--model".to_string(), model.to_string()]);
    }
    args.extend_from_slice(extra_args);
    args
}

/// `PATH`, then the official installer's target, `~/.grok/bin`.
pub(super) fn find_grok() -> Option<PathBuf> {
    find_on_path("grok").or_else(|| grok_fallback(&dirs::home_dir()?))
}

fn grok_fallback(home: &Path) -> Option<PathBuf> {
    let binary = if cfg!(windows) { "grok.exe" } else { "grok" };
    let candidate = home.join(".grok").join("bin").join(binary);
    candidate.is_file().then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grok Build uses the custom-model endpoint for both catalog lookup
    /// and inference. Its auxiliary samplers must follow the selected
    /// model too, rather than asking llmman for Grok's hosted defaults.
    #[test]
    fn grok_env_points_every_model_path_at_llmman() {
        let env = grok_env(
            "docker.io/ai/qwen3.5:0.8b",
            "k",
            "http://127.0.0.1:17434/v1",
            "http://127.0.0.1:17434/v1/models",
            "/tmp/grok/llmman",
        );
        let get = |key| {
            env.iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| *value)
        };
        assert_eq!(
            get("GROK_MODELS_BASE_URL"),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(get("GROK_HOME"), Some("/tmp/grok/llmman"));
        assert_eq!(
            get("GROK_MODELS_LIST_URL"),
            Some("http://127.0.0.1:17434/v1/models")
        );
        assert_eq!(get("GROK_DEFAULT_MODEL"), Some("docker.io/ai/qwen3.5:0.8b"));
        assert_eq!(
            get("GROK_WEB_SEARCH_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(
            get("GROK_SESSION_SUMMARY_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(
            get("GROK_IMAGE_DESCRIPTION_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(
            get("GROK_PROMPT_SUGGESTIONS_MODEL"),
            Some("docker.io/ai/qwen3.5:0.8b")
        );
        assert_eq!(get(GROK_API_KEY_ENV), Some("k"));
        assert_eq!(get("XAI_API_KEY"), Some("k"));
    }

    /// A per-model env_key beats both an existing Grok login and the global
    /// XAI_API_KEY. The generated config contains only the environment
    /// variable's name, never the credential itself.
    #[test]
    fn grok_config_uses_the_model_credential_without_persisting_it() {
        let model = r#"org/model.\"quoted\""#;
        let text = grok_config_document(model, "http://127.0.0.1:17434/v1", &[]);

        let parsed: toml::Value = text.parse().expect("valid TOML");
        let entry = &parsed["model"][model];
        assert_eq!(entry["model"].as_str(), Some(model));
        assert_eq!(
            entry["base_url"].as_str(),
            Some("http://127.0.0.1:17434/v1")
        );
        assert_eq!(entry["env_key"].as_str(), Some(GROK_API_KEY_ENV));
        assert_eq!(entry["api_backend"].as_str(), Some("chat_completions"));
        assert!(entry.get("api_key").is_none());
    }

    #[test]
    fn grok_config_is_written_only_inside_the_isolated_child_home() {
        let root = std::env::temp_dir().join(format!(
            "llmman-grok-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let user_config = root.join("config.toml");
        std::fs::write(&user_config, "[model.mine]\napi_key = \"keep-me\"\n").unwrap();

        let isolated = root.join("llmman");
        write_grok_config(&isolated, "m", "http://127.0.0.1:17434/v1", &[]).unwrap();

        assert_eq!(
            std::fs::read_to_string(user_config).unwrap(),
            "[model.mine]\napi_key = \"keep-me\"\n"
        );
        assert!(std::fs::read_to_string(isolated.join("config.toml"))
            .unwrap()
            .contains("LLMMAN_GROK_API_KEY"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// llmman supplies Grok's model flag unless the caller explicitly
    /// supplied one after `--`; no duplicate flag is handed to the CLI.
    #[test]
    fn grok_args_add_the_model_and_yield_to_an_explicit_override() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            grok_args("m:latest", &args(&["--single", "hi"])),
            ["--model", "m:latest", "--single", "hi"]
        );
        assert_eq!(
            grok_args("m:latest", &args(&["-m", "theirs", "--single", "hi"])),
            ["-m", "theirs", "--single", "hi"]
        );
        assert_eq!(
            grok_args("m:latest", &args(&["--model=theirs"])),
            ["--model=theirs"]
        );
    }

    /// The official installer puts Grok under `~/.grok/bin`, which is
    /// commonly invisible to a non-login process even though the CLI is
    /// installed and usable from the user's shell.
    #[test]
    fn grok_fallback_finds_the_official_installers_target() {
        let home = std::env::temp_dir().join(format!(
            "llmman-grok-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bin = home.join(".grok").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        assert_eq!(grok_fallback(&home), None);
        let grok = bin.join(if cfg!(windows) { "grok.exe" } else { "grok" });
        std::fs::create_dir(&grok).unwrap();
        assert_eq!(grok_fallback(&home), None, "a directory is not a binary");
        std::fs::remove_dir(&grok).unwrap();
        std::fs::write(&grok, "").unwrap();
        assert_eq!(grok_fallback(&home), Some(grok));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn grok_config_lists_the_variants_levels() {
        let text = grok_config_document("m", "http://h/v1", &["none", "high"]);
        let parsed: toml::Value = text.parse().expect("valid TOML");
        assert_eq!(
            parsed["model"]["m"]["reasoning_efforts"],
            toml::Value::Array(vec!["none".into(), "high".into()])
        );
        let text = grok_config_document("m", "http://h/v1", &[]);
        assert!(!text.contains("reasoning_efforts"), "{text}");
    }
}
