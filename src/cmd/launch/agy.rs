//! `llmman launch agy` (Antigravity CLI).

use std::path::{Path, PathBuf};

use anyhow::Context;
use base64::Engine as _;

use super::{exec_with_env, find_on_path, server};

/// AGY speaks Gemini's native generation protocol. The encoded model in the
/// base URL is llmman's routing instruction; AGY also makes auxiliary calls
/// with its own hard-coded model names, so the server deliberately ignores
/// the model segment AGY appends and sends every call to the model selected
/// here.
pub(super) fn launch_agy(model: &str, api_key: &str, extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("agy").ok_or_else(|| anyhow::anyhow!("agy is not installed"))?;
    anyhow::ensure!(
        !extra_args
            .iter()
            .any(|arg| matches!(arg.split('=').next(), Some("--gemini_dir" | "-gemini_dir"))),
        "llmman manages AGY’s --gemini_dir"
    );
    let gemini_dir = agy_settings_dir()?;
    write_agy_settings_at(&gemini_dir)?;
    let mut args = vec![format!("--gemini_dir={}", gemini_dir.display())];
    args.extend_from_slice(extra_args);

    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(model.as_bytes());
    let base_url = format!("{}/gemini/{encoded}", server());
    exec_with_env(
        &bin,
        &args,
        &[
            ("GOOGLE_GEMINI_BASE_URL", base_url.as_str()),
            ("GEMINI_API_KEY", api_key),
            // AGY prefers GOOGLE_API_KEY when both names exist. Override it
            // too so an unrelated key inherited from the shell cannot bypass
            // the credential llmman selected for this request.
            ("GOOGLE_API_KEY", api_key),
        ],
    )
}

pub(super) fn agy_settings_dir() -> anyhow::Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("no home directory")?
        .join(".gemini")
        .join("llmman"))
}

fn write_agy_settings_at(gemini_dir: &Path) -> anyhow::Result<()> {
    let settings_path = gemini_dir.join("antigravity-cli").join("settings.json");
    std::fs::create_dir_all(settings_path.parent().expect("settings file has a parent"))?;
    crate::fsutil::write_atomic(&settings_path, b"{\n  \"modelProvider\": \"gemini\"\n}\n")
        .with_context(|| format!("write {}", settings_path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agy_settings_are_written_to_the_llmman_owned_directory() {
        let dir = std::env::temp_dir().join(format!(
            "llmman-agy-settings-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_agy_settings_at(&dir).unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join("antigravity-cli/settings.json")).unwrap(),
            "{\n  \"modelProvider\": \"gemini\"\n}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
