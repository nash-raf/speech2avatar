# Active task

**Title:** Live streaming architecture — proper lip sync at 1 s chunks (real-time target)
**Spec:** `tasks/001-lipsync-live-diagnosis-and-fix.md`
**State snapshot:** `experiments/2026-04-19-live-state-snapshot.md`
**Deadline:** demo in 2 days
**Owner:** writer (fresh Codex chat)

**Hard constraint:** `chunk_sec = 1.0` is non-negotiable. Training allowed only for minor architectural tweaks and only if truly necessary, with prior approval.

**In scope:**
- P1 (prime suspect) Overlap path silently disabled every chunk — accumulate `_overlap_latents` across chunks.
- P2 Playback scheduler stall — `out_q_size` grows while `play_queue_len=0`, `mode` flips reply↔idle.
- P3 Moshi generation rate `0.67×` real-time — either fix (CUDA stream overlap) or document (Moshi-bound → hardware split).

**Acceptance (summary):**
- `used_overlap_context=True` from chunk 2 onward.
- `boundary_l2` median `< 0.3`, p95 `< 0.8`, max `< 1.5`.
- Browser plays continuously, no reply/idle flicker within a reply.
- `out_q_size` bounded, `play_queue_len > 0` most of the time in a reply.
- Moshi rate ≥ 0.9× real-time, or justified report.

## Status log (newest first)

- [ ] **Writer 3 in progress.** Start from `experiments/001-INTERIM-2026-04-20-baseline-before.md`. Apply P1 accumulator fix, confirm `used_overlap_context=True` from chunk 2, then P2 scheduler, then P3 timing. Use longer audio for final acceptance run (see notes).
- [x] **Writer 2 (2026-04-20):** Baseline debug run completed. Artifacts at `logs/baseline_before_*`. Key finding: boundary stats already healthy (median=0.139, p95=0.230, max=0.278, 0 spikes >1.0). Root: `prev_x0` carry landed in 26ad915 sync. P1 code-level bug still present — `used_overlap_context=False` every chunk. No source edits. No commit.
- [x] **Writer 1 (2026-04-20):** Pod → local sync. Commit: `26ad915`. Interim: `experiments/001-INTERIM-2026-04-20-pod-sync.md`.

Once this task is accepted, move it to `plans/done.md` and pull in `tasks/002-static-head-drift.md`.
