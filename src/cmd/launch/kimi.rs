//! `llmman launch kimi` (Kimi Code CLI).

use super::{exec_with_env, find_on_path, server};

/// kimi: just set OLLAMA_HOST and run the binary. It never gets a model
/// from llmman.
pub(super) fn launch_kimi(extra_args: &[String]) -> anyhow::Result<()> {
    let bin = find_on_path("kimi").ok_or_else(|| anyhow::anyhow!("kimi is not installed"))?;
    let server = server();
    exec_with_env(&bin, extra_args, &[("OLLAMA_HOST", server.as_str())])
}
