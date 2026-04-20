# Workflow (2-day sprint)

This file is the runbook for how we work. Read it before starting any chat.

## Goals (ordered, must-complete before the demo)

1. **Proper live lip sync** — live WebSocket stream lip-syncs cleanly, with no audible/visible chunk-boundary artefacts on a normal conversation. See `tasks/001`.
2. **Static-head mode that actually stays static during speech** — motion latent does not drift the head/upper-face during utterances. See `tasks/002`.

Everything else (post-quant Mimi dataset, iMF branch, Phase 2 re-run) is deferred. See `plans/later.md`.

## Agent roles

- **Architect** = Claude Pro (via Claude Code extension in Cursor). One long-running chat.
  - Reads: `WORKFLOW.md`, `plans/*`, `experiments/*`, occasionally source files.
  - Writes: `plans/active.md`, `plans/backlog.md`, `plans/done.md`, `tasks/NNN-*.md`.
  - **Never** runs commands, never reads raw logs, never edits source code.
  - Goal: keep a small, durable context. Decide what the next task is, write a precise spec, ingest reports.

- **Writer** = Codex Pro (via Codex CLI extension in Cursor). **Fresh chat per task**.
  - Reads: `@.cursor/skills/imtalker-moshi/SKILL.md`, the specific `tasks/NNN-*.md`, relevant code.
  - Writes: source code edits, pod commands, `experiments/YYYY-MM-DD-<slug>.md` final report.
  - Is killed when the task is done. No long-lived writer chats.

- **Composer / inline Cursor agent** = quick one-off edits during review. Also the **browser tool**: it has access to the Playwright / `cursor-ide-browser` MCP and can drive a real browser (load the live WS client page, play audio, screenshot frames, inspect the DOM). Use Composer when you need to *see* the rendered avatar or confirm a seam hiccup visually — Claude Pro (architect) and Codex Pro (writer) do not have browser access in this setup.

## Code sync (no GitHub yet, rsync only)

Fill in these once in your shell, then only two commands to remember.

```bash
# Edit ~/.bashrc or ~/.zshrc  (current pod: root@74.15.1.150 port 30998)
export POD_HOST=root@74.15.1.150
export POD_PORT=30998
export POD_KEY=$HOME/.ssh/id_ed25519
export POD_IMT=/workspace/IMTalker
```

### Push local -> pod (do this every time the writer finishes editing)

```bash
rsync -avz --delete \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
  --exclude 'checkpoints/' --exclude 'exps/' --exclude '*.ckpt' --exclude '*.pt' \
  --exclude 'logs/' --exclude 'live_ws_debug.log' --exclude 'dump_reply/' \
  --exclude '*.mp4' --exclude '*.png' --exclude '*.jpg' --exclude '.venv/' \
  --exclude 'web_vendor/' --exclude 'assets/' \
  -e "ssh -p $POD_PORT -i $POD_KEY" \
  /home/user/D/working_yay/both/IMTalker/ \
  $POD_HOST:$POD_IMT/
```

### Pull pod -> local (for logs, metrics, debug dumps after a run)

```bash
rsync -avz \
  -e "ssh -p $POD_PORT -i $POD_KEY" \
  $POD_HOST:$POD_IMT/live_ws_debug.log \
  $POD_HOST:$POD_IMT/dump_reply/ \
  /home/user/D/working_yay/both/IMTalker/logs/
```

Never paste raw logs into chat. Always use `pod_bin/boundary_summary.py` first (see below).

## Pod helper scripts (deploy once, use forever)

Scripts live locally in `pod_bin/`. Copy them to the pod with:

```bash
rsync -avz -e "ssh -p $POD_PORT -i $POD_KEY" pod_bin/ $POD_HOST:/workspace/bin/
ssh -p $POD_PORT -i $POD_KEY $POD_HOST 'chmod +x /workspace/bin/*'
```

| Script | Purpose | Example |
|---|---|---|
| `boundary_summary.py` | Parse `live_ws_debug.log`, print per-chunk boundary_l2 stats | `ssh pod '/workspace/bin/boundary_summary.py /workspace/IMTalker/live_ws_debug.log'` |
| `tail_log.sh`         | Print last N lines of any file (default 80) | `ssh pod '/workspace/bin/tail_log.sh /workspace/IMTalker/live_ws_debug.log 120'` |
| `sync.sh`             | Tiny wrapper that prints the rsync one-liner you need to run locally | `./pod_bin/sync.sh` |

## Per-task loop (follow this exactly)

1. **Architect session (5–10 min)** — write `tasks/NNN-<slug>.md`. Update `plans/active.md`. Commit locally.
2. **Writer session (30 min – 3 h)** — fresh Codex chat. Prompt it with the template in `PROMPTS.md`. It edits code, rsyncs to pod, runs tests, writes `experiments/YYYY-MM-DD-<slug>.md`.
3. **Architect ingestion (5 min)** — architect reads the experiment file, updates `plans/active.md` to the next task, appends one-line to `plans/done.md`.
4. **Kill the writer chat.** Start a fresh one for the next task.

## When to escalate to Composer (browser)

Writer agents (Codex) can reproduce boundary issues numerically via `boundary_summary.py` and eyeball the saved mp4, but they can NOT drive a live browser to confirm the real-time playback looks right.

Escalate to **Cursor Composer** when you need to:
- Load the live WS client page (`launch_live_ws.py` serves it) and confirm audio+video actually stream.
- Screenshot specific frames to eyeball mouth shapes during a reply.
- Compare two live sessions (static vs non-static head) side by side in real time.
- Inspect the DOM / console for client-side errors during a reply.

Workflow when this happens:
1. Writer (Codex) writes an interim note in `experiments/NNN-INTERIM-<timestamp>.md` explaining what needs browser verification and what to look for.
2. User opens Cursor Composer. Prompt: `Read @experiments/NNN-INTERIM-<timestamp>.md. Drive the live WS client at <pod_url> via Playwright and verify the items listed. Append your findings (screenshots + verdict) to the same file.`
3. User returns control to the writer Codex chat, which continues from the updated interim file.

## Hard rules

- **Every chat starts with** `Read @.cursor/skills/imtalker-moshi/SKILL.md` and `Read WORKFLOW.md`. Do not re-explain the project.
- **Writer chat gets slow or >30 tool calls?** Stop. Write an interim report to `experiments/`. Start a fresh chat pointing at that report.
- **Logs never enter chat context raw.** Summarize with `boundary_summary.py` or similar first.
- **Every task ends with** a report in `experiments/` AND a `git commit` locally.
- **Architect writes no code, runs no commands.** If you catch yourself wanting to, open a writer chat instead.
- **`chunk_sec = 1.0` is NON-NEGOTIABLE.** Real-time live streaming is the target.
- **Training is allowed only for minor architecture tweaks, and strongly preferred NOT.** If a writer thinks a fix needs retraining, it must stop and ask the user for approval before starting training.
- **Writers may add new `[DBG/...]` prints** when it helps isolate a bug. Gate them on `self.debug_session`, use consistent prefixes, one grep-able line per event. If a new tag is added, update `pod_bin/boundary_summary.py` so future runs summarise it automatically.
