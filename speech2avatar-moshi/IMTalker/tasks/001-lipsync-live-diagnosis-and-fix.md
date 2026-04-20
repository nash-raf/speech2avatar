# Task 001: Live streaming architecture — proper lip sync at 1 s chunks

## Context (read before editing)

- `@.cursor/skills/imtalker-moshi/SKILL.md`
- `@WORKFLOW.md`
- `@experiments/2026-04-19-live-state-snapshot.md` ← the current debug state + diagnosis
- `@runbooks/live-launch.md` ← exact debug launch recipe

## Hard constraint

- **`chunk_sec = 1.0` is NON-NEGOTIABLE.** Do not propose increasing chunk_sec to work around seams or latency. The whole point of this project is real-time live streaming.
- Training is **allowed for small architecture tweaks but strongly preferred not**. Try code-only fixes first. If a minor retrain is the only path, flag it in the report and wait for approval before starting training.

## The specific problems to fix (ordered by suspected impact)

### P1. Overlap/context path is silently disabled every chunk (prime suspect for boundary spikes)

- Offline is stable. Live has `boundary_l2` spikes of 2–7 during speech.
- Live debug logs show `used_overlap_context=False` and `overlap_frames_used=0` on **every** chunk.
- Already diagnosed in the snapshot: `live_pipeline.py:655` does `self._overlap_latents = latents[-N:].clone()`, but a 1-second chunk only produces 12–13 raw Mimi latents and `overlap_context_latent_count=24`. The gate on line 617 (`>= 24`) never passes.
- **Fix:** accumulate `_overlap_latents` across chunks instead of overwriting. Minimum-change shape:
  ```python
  new_tail = torch.cat([self._overlap_latents, latents], dim=0) if self._overlap_latents is not None else latents
  self._overlap_latents = new_tail[-self.overlap_context_latent_count:].clone()
  ```
  Keep semantics: first chunk `_overlap_latents=None` → gate fails, no-op. From chunk 2 onward, gate should pass and `latents_for_align` should be `cat([_overlap_latents, latents])` as already coded.
- Double-check reset points: `reset_reply()` (line ~879), warmup (~461), `__init__` (~281) — these already set to None, fine. Do not add new resets.
- Verify `overlap_frames_used > 0` in fresh logs and `boundary_l2` improves.

### P2. Playback scheduler stall — `out_q_size` growing while `play_queue_len=0`, `mode` flipping reply↔idle

- Producer/consumer mismatch between render output and client play queue. Logs show packets piling up in `out_q` while the play queue stays empty, and the client keeps flipping between `reply` and `idle`.
- Suspects: `launch_live_ws.py` and/or `live_pipeline_ws.py`:
  - Does the WS send task block when the client hasn't signalled a ready buffer?
  - Is the `reply/idle` mode driven by a timer that times out too aggressively? (If a chunk is ~1s, and Moshi is 0.67× realtime, the play queue will go empty between chunks — the UI should NOT flip to idle in that gap.)
  - Is there a drop-if-stale policy that discards chunks older than some threshold?
- **Diagnostic approach:**
  - Add a few targeted prints (see "Adding new debugs" below) at: (i) where rendered frames leave `out_q`, (ii) where they land in `play_queue`, (iii) where `mode` transitions happen. Log per-event: `chunk_age`, `out_q_size`, `play_queue_len`, `mode`, wall-clock delta since last chunk.
  - Rerun and read the summary.
- **Expected fix shape (don't commit until diagnosed):**
  - Widen the mode-flip debouncing so short gaps at 1s chunk cadence don't flip to idle.
  - Make sure the sender is draining `out_q` into `play_queue` continuously rather than waiting on a client event.
  - Consider a small lead buffer (e.g., 1 chunk) before declaring `reply` active so the client always has at least one chunk queued.

### P3. Moshi 0.67× real-time

- `generated=22.00s wall=32.75s rate=0.67x` is below real-time. Unless this is GPU contention, the entire stream cannot be real-time regardless of everything else.
- **Diagnose:**
  - Log per-call timings in the Moshi generate loop vs the Mimi encode path vs IMTalker render. Which one is eating wall clock?
  - Check whether Moshi LM + Mimi encoder + IMTalker renderer are sharing one CUDA stream. Confirm `render_stream = torch.cuda.Stream()` is distinct from Moshi's stream and that they actually overlap.
  - Check GPU utilisation during a reply (`nvidia-smi dmon` for a few seconds). If util ≈ 100% already, contention is the cause and the fix is hardware (add a second GPU) or move IMTalker render off the Moshi GPU.
- **Allowed fixes (preferred order):**
  1. Ensure IMTalker render really runs on its own stream concurrently with Moshi.
  2. If Moshi LM alone is the bottleneck, note it in the report and propose a hardware split (two GPUs) as a follow-up — do not try to quantise or swap the Moshi model in this task.
  3. Do NOT reduce `nfe` below 5 or raise above 8.
  4. Do NOT disable the renderer.

## Pod-vs-local code divergence (resolve FIRST)

The pod already has the `prev_x0` carry fix and a few other recent edits; local `FM.py` does not. Before editing:
1. rsync **pod -> local** (full `generator/`, `live_pipeline.py`, `launch_live*.py`) so local is the source of truth going forward.
2. Commit this pull locally with message: `sync: pull recent pod-side fixes into local (prev_x0 carry, ...)`.
3. Then do all further edits locally and push to pod with the usual rsync.

## Adding new debug prints (encouraged when useful)

You may add targeted `if self.debug_session: print("[DBG/...] ...")` lines anywhere that helps isolate the bug. Rules:
- Gate them behind `self.debug_session` so normal runs stay clean.
- Prefix consistently: `[DBG/overlap]`, `[DBG/queue]`, `[DBG/moshi]`, `[DBG/render]`, `[DBG/x0]`.
- Keep each line grep-able (one line, key=value pairs).
- If you add a new tag, update `pod_bin/boundary_summary.py` (in local `pod_bin/` first, then rsync to pod) so future runs summarise it automatically.
- Remove purely exploratory prints before the final report, keep only the ones that should live long-term.

## Acceptance

On `source_5 + audio_3` live WS run, with `--debug_session`.

**Ref path (confirmed on pod):** `--ref_path /workspace/IMTalker/assets/source_5.png`
(`/workspace/sources/source_5.png` does not exist — use the assets path for all reruns.)

**Audio length:** `audio_3.wav` on pod is 14.68 s (15 chunks). For the 30 s acceptance run, loop or extend it:

```bash
ffmpeg -stream_loop 2 -i /workspace/IMTalker/assets/audio_3.wav -c copy /tmp/audio_3_30s.wav
```

Feed `/tmp/audio_3_30s.wav` to the driver for the final acceptance run.

- `used_overlap_context=True` and `overlap_frames_used > 0` from chunk 2 onward.
- `boundary_l2` median `< 0.3`, p95 `< 0.8`, max `< 1.5` across a 30 s reply.
- Browser plays continuously with no visible `reply/idle` flicker during a single reply.
- `out_q_size` stays bounded (≤10 in steady state); `play_queue_len` is non-zero most of the time during a reply.
- Moshi rate ≥ 0.9× real-time OR a written justification in the report that the bottleneck is confirmed to be Moshi LM itself and needs a hardware split.

## Steps

1. rsync pod → local. Commit locally.
2. Reproduce with debug on pod. Pull logs. Run `boundary_summary.py`.
3. Fix P1 (overlap accumulation). rsync local → pod. Rerun. Confirm `used_overlap_context=True` and boundary stats improve.
4. Diagnose P2 with added debug prints. Propose and apply the minimal orchestration fix. Verify mode/queue sanity in the logs.
5. Diagnose P3. Either fix (CUDA stream overlap) or document (Moshi LM bottleneck → needs hardware split). Either way, include per-component timing numbers in the report.
6. Final live run on `source_5 + audio_3`. Capture to mp4.
7. Write `experiments/YYYY-MM-DD-001-live-streaming-architecture.md` with:
   - Commit hashes.
   - file:line cites of every edit.
   - Before/after for the `boundary_summary.py` output.
   - Before/after `used_overlap_context` ratio.
   - Per-component timing table (Moshi LM, Mimi encode, FM sample, render, encode/send).
   - Browser behaviour (reply/idle transitions, play_queue_len distribution).
   - What actually moved each acceptance metric, what didn't, any remaining risks.
8. Local git commit all changes.

## Do not

- Do not change `chunk_sec`.
- Do not touch the renderer weights or architecture.
- Do not touch the iMF branch.
- Do not retrain the generator unless a minor architectural tweak genuinely requires it, and only after flagging in the report and waiting for approval.
- Do not refactor the threading model wholesale; fix it surgically.
- Do not paste raw logs into the chat; always summarise first with `pod_bin/boundary_summary.py`.
