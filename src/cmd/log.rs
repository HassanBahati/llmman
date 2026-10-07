//! `llmman log` — `git log` for the prompts `llmman serve` has seen. Reads
//! `crate::promptlog`'s file directly (no daemon, as `git log` needs no
//! server), newest first, through a pager on a terminal.

use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context};
use chrono::{DateTime, Local, Utc};
use clap::Args;

use crate::promptlog::Entry;

#[derive(Args, Debug)]
pub struct LogArgs {
    /// Print the selected history, then stream new prompts without a pager.
    /// --max-count, --skip and --reverse apply only to the initial history
    #[arg(short = 'f', long)]
    pub follow: bool,
    /// Limit the number of prompts to output
    #[arg(short = 'n', long = "max-count", value_name = "NUMBER")]
    pub max_count: Option<usize>,
    /// Skip NUMBER prompts before starting to show the output
    #[arg(long, value_name = "NUMBER", default_value_t = 0)]
    pub skip: usize,
    /// Show prompts more recent than a specific date (RFC 3339, YYYY-MM-DD,
    /// "yesterday", or "2 hours ago")
    #[arg(long, visible_alias = "after", value_name = "DATE")]
    pub since: Option<String>,
    /// Show prompts older than a specific date
    #[arg(long, visible_alias = "before", value_name = "DATE")]
    pub until: Option<String>,
    /// Limit to prompts sent to a model matching the pattern (regular
    /// expression). Repeat to match any of several
    #[arg(long, value_name = "PATTERN")]
    pub model: Vec<String>,
    /// Limit to prompts whose text matches the pattern (regular
    /// expression). Repeat to match any of several
    #[arg(long, value_name = "PATTERN")]
    pub grep: Vec<String>,
    /// Match the limiting patterns without regard to letter case
    #[arg(short = 'i', long = "regexp-ignore-case")]
    pub ignore_case: bool,
    /// One line per prompt: its abbreviated id and first line
    #[arg(long)]
    pub oneline: bool,
    /// Oldest first
    #[arg(long)]
    pub reverse: bool,
    /// Do not pipe the output into a pager
    #[arg(long)]
    pub no_pager: bool,
}

/// git's `log -3` shorthand for `-n 3`, which clap has no spelling for:
/// a bare `-<digits>` after `log` is rewritten before parsing.
pub fn expand_count_shorthand<I: IntoIterator<Item = std::ffi::OsString>>(
    args: I,
) -> Vec<std::ffi::OsString> {
    let mut args: Vec<std::ffi::OsString> = args.into_iter().collect();
    if args.get(1).is_some_and(|a| a == "log") {
        for arg in &mut args[2..] {
            if let Some(n) = arg.to_str().and_then(|a| a.strip_prefix('-')) {
                if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) {
                    *arg = format!("-n{n}").into();
                }
            }
        }
    }
    args
}

pub fn run(args: &LogArgs) -> anyhow::Result<()> {
    let path = crate::promptlog::path()?;
    let in_window = window(args.since.as_deref(), args.until.as_deref(), Utc::now())?;
    let model = patterns(&args.model, args.ignore_case)?;
    let grep = patterns(&args.grep, args.ignore_case)?;
    let matches = |e: &Entry| {
        in_window(&e.time)
            && model.as_ref().is_none_or(|m| m.is_match(&e.model))
            && grep.as_ref().is_none_or(|g| g.is_match(&e.prompt))
    };
    if args.follow {
        return follow(&path, args, matches);
    }
    let mut entries = crate::promptlog::read(&path)
        .with_context(|| format!("read prompt log {}", path.display()))?;

    entries.retain(matches);

    let color = io::stdout().is_terminal();
    let out = render(
        &select(&entries, args.skip, args.max_count, args.reverse),
        args,
        color,
        &mut false,
    );
    crate::pager::emit("log", &out, !args.no_pager)
}

fn render(entries: &[&Entry], args: &LogArgs, color: bool, has_output: &mut bool) -> String {
    let mut out = String::new();
    for entry in entries {
        if args.oneline {
            oneline(entry, color, &mut out);
        } else {
            if *has_output {
                out.push('\n');
            }
            full(entry, color, &mut out);
        }
        *has_output = true;
    }
    out
}

fn follow(path: &Path, args: &LogArgs, matches: impl Fn(&Entry) -> bool) -> anyhow::Result<()> {
    tokio::runtime::Runtime::new()?.block_on(async {
        let mut tail = Tail::default();
        let mut entries = tail.read(path)?;
        entries.retain(&matches);
        let color = io::stdout().is_terminal();
        let mut has_output = false;
        let out = render(
            &select(&entries, args.skip, args.max_count, args.reverse),
            args,
            color,
            &mut has_output,
        );
        if !write_follow(&out)? {
            return Ok(());
        }
        let interrupt = tokio::signal::ctrl_c();
        tokio::pin!(interrupt);
        loop {
            tokio::select! {
                result = &mut interrupt => {
                    result?;
                    return Ok(());
                }
                _ = tokio::time::sleep(Duration::from_millis(250)) => {
                    let entries = tail.read(path)?;
                    let shown: Vec<_> = entries.iter().filter(|entry| matches(entry)).collect();
                    let out = render(&shown, args, color, &mut has_output);
                    if !write_follow(&out)? {
                        return Ok(());
                    }
                }
            }
        }
    })
}

fn write_follow(text: &str) -> io::Result<bool> {
    if text.is_empty() {
        return Ok(true);
    }
    let mut stdout = io::stdout().lock();
    match stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(false),
        Err(e) => Err(e),
    }
}

#[derive(Default)]
struct Tail {
    reader: Option<BufReader<File>>,
    pending: Vec<u8>,
    checkpoint: Vec<u8>,
}

impl Tail {
    fn read(&mut self, path: &Path) -> io::Result<Vec<Entry>> {
        if self.reader.is_none() {
            match File::open(path) {
                Ok(file) => self.reader = Some(BufReader::new(file)),
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
                Err(e) => return Err(e),
            }
        }
        let reader = self.reader.as_mut().unwrap();
        let position = reader.stream_position()?;
        let mut reset = reader.get_ref().metadata()?.len() < position;
        // The log is append-only. A bounded checkpoint also detects refills
        // without rereading the entire history while idle.
        if !reset && !self.checkpoint.is_empty() {
            reader.seek(SeekFrom::Start(position - self.checkpoint.len() as u64))?;
            let mut bytes = vec![0; self.checkpoint.len()];
            match reader.read_exact(&mut bytes) {
                Ok(()) => reset = bytes != self.checkpoint,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => reset = true,
                Err(e) => return Err(e),
            }
            reader.seek(SeekFrom::Start(position))?;
        }
        if reset {
            reader.seek(SeekFrom::Start(0))?;
            self.pending.clear();
            self.checkpoint.clear();
        }
        let mut entries = Vec::new();
        loop {
            let read = reader.read_until(b'\n', &mut self.pending)?;
            if read == 0 {
                break;
            }
            let bytes = &self.pending[self.pending.len() - read..];
            const CHECKPOINT_SIZE: usize = 4096;
            if bytes.len() >= CHECKPOINT_SIZE {
                self.checkpoint.clear();
                self.checkpoint
                    .extend_from_slice(&bytes[bytes.len() - CHECKPOINT_SIZE..]);
            } else {
                let excess = (self.checkpoint.len() + bytes.len()).saturating_sub(CHECKPOINT_SIZE);
                self.checkpoint.drain(..excess);
                self.checkpoint.extend_from_slice(bytes);
            }
            // A JSON value can parse before its writer has finished the line.
            // Keep the partial line across polls and emit only once it is complete.
            if self.pending.last() == Some(&b'\n') {
                if let Ok(entry) = serde_json::from_slice(&self.pending) {
                    entries.push(entry);
                }
                self.pending.clear();
            }
        }
        Ok(entries)
    }
}

/// As git: `--skip`/`-n` select newest-first, `--reverse` then flips
/// only what's shown (`log --reverse -3` is the newest three, oldest
/// first).
fn select(entries: &[Entry], skip: usize, max: Option<usize>, reverse: bool) -> Vec<&Entry> {
    let mut shown: Vec<&Entry> = entries
        .iter()
        .rev()
        .skip(skip)
        .take(max.unwrap_or(usize::MAX))
        .collect();
    if reverse {
        shown.reverse();
    }
    shown
}

/// Whether an RFC 3339 time is within `--since`/`--until`.
pub(super) fn window(
    since: Option<&str>,
    until: Option<&str>,
    now: DateTime<Utc>,
) -> anyhow::Result<impl Fn(&str) -> bool> {
    let since = since.map(|s| parse_date(s, now)).transpose()?;
    let until = until.map(|s| parse_date(s, now)).transpose()?;
    Ok(move |time: &str| {
        let time = DateTime::parse_from_rfc3339(time).ok();
        since.is_none_or(|s| time.is_some_and(|t| t >= s))
            && until.is_none_or(|u| time.is_some_and(|t| t <= u))
    })
}

/// Several patterns match when any does, as in git; none is no filter.
pub(super) fn patterns(
    patterns: &[String],
    ignore_case: bool,
) -> anyhow::Result<Option<regex::RegexSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    regex::RegexSetBuilder::new(patterns)
        .case_insensitive(ignore_case)
        .build()
        .map(Some)
        .context("invalid pattern")
}

/// RFC 3339; local `YYYY-MM-DD` (midnight) or `YYYY-MM-DD HH:MM[:SS]`;
/// `yesterday`; git's `N <unit>[s] ago` (also `N.<unit>.ago`).
fn parse_date(s: &str, now: DateTime<Utc>) -> anyhow::Result<DateTime<Utc>> {
    let s = s.trim();
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Ok(t.with_timezone(&Utc));
    }
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"] {
        if let Ok(t) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return local_to_utc(t);
        }
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return local_to_utc(d.and_hms_opt(0, 0, 0).unwrap());
    }
    let words: Vec<&str> = s.split(['.', ' ']).filter(|w| !w.is_empty()).collect();
    let (n, unit): (i64, &str) = match words[..] {
        ["yesterday"] => (1, "day"),
        [n, unit, "ago"] => (n.parse().ok().unwrap_or(-1), unit.trim_end_matches('s')),
        _ => bail!("invalid date: {s:?}"),
    };
    let seconds: i64 = match unit {
        "second" | "sec" => 1,
        "minute" | "min" => 60,
        "hour" => 3600,
        "day" => 86400,
        "week" => 86400 * 7,
        "month" => 86400 * 30,
        "year" => 86400 * 365,
        _ => bail!("invalid date: {s:?}"),
    };
    n.checked_mul(seconds)
        .filter(|_| n >= 0)
        .and_then(chrono::Duration::try_seconds)
        .and_then(|d| now.checked_sub_signed(d))
        .with_context(|| format!("invalid date: {s:?}"))
}

fn local_to_utc(t: chrono::NaiveDateTime) -> anyhow::Result<DateTime<Utc>> {
    t.and_local_timezone(Local)
        .single()
        .map(|t| t.with_timezone(&Utc))
        .ok_or_else(|| anyhow::anyhow!("not a valid local time: {t}"))
}

fn paint(text: &str, color: bool) -> String {
    crate::fmt::paint(text, crate::fmt::YELLOW, color)
}

/// Request-supplied text with control characters (tab aside) dropped:
/// no driving the terminal — `less -R` passes escapes through — and no
/// forged header lines from a newline in a model name.
pub(super) fn clean(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() || *c == '\t')
        .collect()
}

/// `git log`'s layout: header, blank line, message indented four.
fn full(e: &Entry, color: bool, out: &mut String) {
    out.push_str(&paint(&format!("prompt {}", e.id), color));
    out.push('\n');
    out.push_str(&format!("Model:  {}\n", clean(&e.model)));
    if let Some(client) = &e.client {
        out.push_str(&format!("Client: {}\n", clean(client)));
    }
    out.push_str(&format!("Route:  {}\n", e.route));
    out.push_str(&format!("Date:   {}\n\n", git_date(&e.time)));
    for line in e.prompt.lines() {
        out.push_str("    ");
        out.push_str(&clean(line));
        out.push('\n');
    }
}

fn oneline(e: &Entry, color: bool, out: &mut String) {
    out.push_str(&paint(&crate::fmt::short_id(&e.id), color));
    out.push(' ');
    out.push_str(&clean(e.prompt.lines().next().unwrap_or_default()));
    out.push('\n');
}

/// git's date format, local time: `Sun Sep 6 12:34:56 2026 +0100`.
fn git_date(rfc3339: &str) -> String {
    match DateTime::parse_from_rfc3339(rfc3339) {
        Ok(t) => t
            .with_timezone(&Local)
            .format("%a %b %-d %H:%M:%S %Y %z")
            .to_string(),
        Err(_) => rfc3339.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> Entry {
        Entry {
            id: "0123456789abcdef0123456789abcdef01234567".into(),
            time: "2026-09-06T11:34:56Z".into(),
            route: "/v1/chat/completions".into(),
            model: "qwen3:8b".into(),
            client: Some("claude-cli/1.2.3".into()),
            prompt: "fix the failing test\nthen commit".into(),
        }
    }

    #[test]
    fn follow_reads_each_complete_line_once_and_recovers_after_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.jsonl");
        let mut tail = Tail::default();
        assert!(tail.read(&path).unwrap().is_empty());
        let first = entry();
        crate::promptlog::append(&path, &first).unwrap();
        assert_eq!(tail.read(&path).unwrap(), [first]);
        assert!(tail.read(&path).unwrap().is_empty());
        let second = Entry {
            id: "second".into(),
            ..entry()
        };
        let bytes = serde_json::to_vec(&second).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        assert!(tail.read(&path).unwrap().is_empty());
        file.write_all(b"\ninvalid json\n").unwrap();
        assert_eq!(tail.read(&path).unwrap(), [second]);
        assert!(tail.read(&path).unwrap().is_empty());
        File::create(&path).unwrap();
        assert!(tail.read(&path).unwrap().is_empty());
        crate::promptlog::append(&path, &entry()).unwrap();
        assert_eq!(tail.read(&path).unwrap(), [entry()]);
    }

    #[test]
    fn follow_recovers_when_truncation_and_refill_happen_between_polls() {
        for extra in ["", " with a longer prompt"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("prompts.jsonl");
            let mut tail = Tail::default();
            let first = entry();
            crate::promptlog::append(&path, &first).unwrap();
            assert_eq!(tail.read(&path).unwrap(), [first]);
            let replacement = Entry {
                id: "fedcba9876543210fedcba9876543210fedcba98".into(),
                prompt: format!("{}{extra}", entry().prompt),
                ..entry()
            };
            File::create(&path).unwrap();
            crate::promptlog::append(&path, &replacement).unwrap();
            assert_eq!(tail.read(&path).unwrap(), [replacement]);
            assert!(tail.read(&path).unwrap().is_empty());
            crate::promptlog::append(&path, &entry()).unwrap();
            assert_eq!(tail.read(&path).unwrap(), [entry()]);
        }
    }

    #[test]
    fn follow_detects_a_refill_that_preserves_the_last_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.jsonl");
        let mut tail = Tail::default();
        let first = entry();
        let last = Entry {
            id: "last".into(),
            ..entry()
        };
        crate::promptlog::append(&path, &first).unwrap();
        crate::promptlog::append(&path, &last).unwrap();
        assert_eq!(tail.read(&path).unwrap(), [first, last.clone()]);
        let previous_len = std::fs::metadata(&path).unwrap().len();
        let replacement = Entry {
            id: "fedcba9876543210fedcba9876543210fedcba98".into(),
            ..entry()
        };
        File::create(&path).unwrap();
        crate::promptlog::append(&path, &replacement).unwrap();
        crate::promptlog::append(&path, &last).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), previous_len);
        assert_eq!(tail.read(&path).unwrap(), [replacement, last]);
        assert!(tail.read(&path).unwrap().is_empty());
    }

    #[test]
    fn follow_keeps_a_bounded_checkpoint_for_large_entries_and_appends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.jsonl");
        let mut tail = Tail::default();
        let large = Entry {
            prompt: "x".repeat(16384),
            ..entry()
        };
        crate::promptlog::append(&path, &large).unwrap();
        assert_eq!(tail.read(&path).unwrap(), [large]);
        for _ in 0..3 {
            assert_eq!(tail.checkpoint.len(), 4096);
            assert!(tail.read(&path).unwrap().is_empty());
            crate::promptlog::append(&path, &entry()).unwrap();
            assert_eq!(tail.read(&path).unwrap(), [entry()]);
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(tail.checkpoint, bytes[bytes.len() - 4096..]);
        }
    }

    #[test]
    fn follow_discards_a_partial_line_when_the_file_is_refilled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.jsonl");
        std::fs::write(&path, b"{\"id\":\"unfinished").unwrap();
        let mut tail = Tail::default();
        assert!(tail.read(&path).unwrap().is_empty());
        File::create(&path).unwrap();
        crate::promptlog::append(&path, &entry()).unwrap();
        assert_eq!(tail.read(&path).unwrap(), [entry()]);
        assert!(tail.read(&path).unwrap().is_empty());
    }

    #[test]
    fn full_format_mirrors_git_log() {
        let mut out = String::new();
        full(&entry(), false, &mut out);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "prompt 0123456789abcdef0123456789abcdef01234567");
        assert_eq!(lines[1], "Model:  qwen3:8b");
        assert_eq!(lines[2], "Client: claude-cli/1.2.3");
        assert_eq!(lines[3], "Route:  /v1/chat/completions");
        assert!(lines[4].starts_with("Date:   "), "{}", lines[4]);
        assert_eq!(lines[5], "");
        assert_eq!(lines[6], "    fix the failing test");
        assert_eq!(lines[7], "    then commit");
        assert_eq!(lines.len(), 8);
    }

    #[test]
    fn the_client_line_is_left_out_when_unknown() {
        let mut out = String::new();
        full(
            &Entry {
                client: None,
                ..entry()
            },
            false,
            &mut out,
        );
        assert!(!out.contains("Client:"));
    }

    #[test]
    fn oneline_is_the_short_id_and_first_line() {
        let mut out = String::new();
        oneline(&entry(), false, &mut out);
        assert_eq!(out, "0123456789ab fix the failing test\n");
    }

    #[test]
    fn color_wraps_only_the_header_in_yellow() {
        let mut out = String::new();
        full(&entry(), true, &mut out);
        assert!(out.starts_with("\x1b[33mprompt 0123456789abcdef0123456789abcdef01234567\x1b[0m\n"));
        assert_eq!(out.matches("\x1b[").count(), 2);
    }

    #[test]
    fn request_supplied_text_cannot_drive_the_terminal() {
        let hostile = Entry {
            model: "m\x1b[2J\nDate:   forged".into(),
            client: Some("c\u{9b}0m\r".into()),
            prompt: "\x1b]0;title\x07line one\n\tindented\x08".into(),
            ..entry()
        };
        let mut out = String::new();
        full(&hostile, false, &mut out);
        assert!(!out.contains('\x1b') && !out.contains('\u{9b}') && !out.contains('\x07'));
        assert!(out.contains("Model:  m[2JDate:   forged\n"));
        assert!(out.contains("Client: c0m\n"));
        assert!(out.contains("    ]0;titleline one\n    \tindented\n"));
        let mut one = String::new();
        oneline(&hostile, false, &mut one);
        assert_eq!(one, "0123456789ab ]0;titleline one\n");
    }

    #[test]
    fn git_date_uses_gits_layout() {
        // Local zone, so check the shape rather than the hour.
        let d = git_date("2026-09-06T11:34:56Z");
        let parts: Vec<&str> = d.split(' ').collect();
        assert_eq!(parts.len(), 6, "{d}");
        assert_eq!(parts[1], "Sep");
        assert_eq!(parts[3].len(), 8);
        assert_eq!(parts[4], "2026");
        assert!(parts[5].starts_with('+') || parts[5].starts_with('-'));
        assert_eq!(git_date("garbage"), "garbage");
    }

    #[test]
    fn relative_dates_count_back_from_now() {
        let now = DateTime::parse_from_rfc3339("2026-09-06T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let at = |s: &str| parse_date(s, now).unwrap().to_rfc3339();
        assert_eq!(at("2 hours ago"), "2026-09-06T10:00:00+00:00");
        assert_eq!(at("1.week.ago"), "2026-08-30T12:00:00+00:00");
        assert_eq!(at("yesterday"), "2026-09-05T12:00:00+00:00");
        assert_eq!(at("30 minutes ago"), "2026-09-06T11:30:00+00:00");
        assert_eq!(at("2026-09-01T00:00:00Z"), "2026-09-01T00:00:00+00:00");
        for bad in [
            "fortnight",
            "3 fortnights ago",
            "-2 days ago",
            "99999999999999 years ago",
        ] {
            assert!(parse_date(bad, now).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_plain_date_is_local_midnight() {
        let now = Utc::now();
        let t = parse_date("2026-09-06", now).unwrap().with_timezone(&Local);
        assert_eq!(
            t.format("%Y-%m-%d %H:%M:%S").to_string(),
            "2026-09-06 00:00:00"
        );
    }

    #[test]
    fn reverse_flips_the_selection_not_the_whole_log() {
        let entries: Vec<Entry> = (1..=5)
            .map(|i| Entry {
                prompt: format!("p{i}"),
                ..entry()
            })
            .collect();
        fn prompts(shown: Vec<&Entry>) -> Vec<&str> {
            shown.iter().map(|e| e.prompt.as_str()).collect()
        }
        assert_eq!(
            prompts(select(&entries, 0, None, false)),
            ["p5", "p4", "p3", "p2", "p1"]
        );
        assert_eq!(
            prompts(select(&entries, 0, Some(3), false)),
            ["p5", "p4", "p3"]
        );
        assert_eq!(
            prompts(select(&entries, 0, Some(3), true)),
            ["p3", "p4", "p5"]
        );
        assert_eq!(prompts(select(&entries, 1, Some(2), true)), ["p3", "p4"]);
    }

    #[test]
    fn a_bare_dash_number_after_log_becomes_max_count() {
        let expand = |args: &[&str]| -> Vec<String> {
            expand_count_shorthand(args.iter().map(std::ffi::OsString::from))
                .into_iter()
                .map(|a| a.into_string().unwrap())
                .collect()
        };
        assert_eq!(
            expand(&["llmman", "log", "-3", "--oneline"]),
            ["llmman", "log", "-n3", "--oneline"]
        );
        assert_eq!(
            expand(&["llmman", "log", "-n", "3"]),
            ["llmman", "log", "-n", "3"]
        );
        assert_eq!(expand(&["llmman", "log", "-"]), ["llmman", "log", "-"]);
        assert_eq!(expand(&["llmman", "ps", "-3"]), ["llmman", "ps", "-3"]);
    }

    #[test]
    fn several_patterns_match_any_and_ignore_case_applies_to_all() {
        let re = patterns(&["foo".into(), "bar".into()], true)
            .unwrap()
            .unwrap();
        assert!(re.is_match("FOO"));
        assert!(re.is_match("a Bar"));
        assert!(!re.is_match("baz"));
        // Independent patterns: a capture name may repeat across them.
        assert!(patterns(&["(?P<x>a)".into(), "(?P<x>b)".into()], false).is_ok());
        assert!(patterns(&[], false).unwrap().is_none());
        assert!(patterns(&["(".into()], false).is_err());
    }
}
