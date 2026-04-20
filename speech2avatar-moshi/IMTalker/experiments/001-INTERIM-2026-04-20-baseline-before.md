# Task 001 interim: baseline debug run completed

Date: 2026-04-20
Writer state: stopped because this chat reached the workflow tool-call budget after completing the before/P1 baseline run.

## What was read

Read in the user-requested order:

1. `/home/user/D/.cursor/skills/imtalker-moshi/SKILL.md`
2. `WORKFLOW.md`
3. `experiments/001-INTERIM-2026-04-20-pod-sync.md`
4. `experiments/2026-04-19-live-state-snapshot.md`
5. `tasks/001-lipsync-live-diagnosis-and-fix.md`
6. `runbooks/live-launch.md`

Confirmed local git log starts with:

```text
26ad915 sync: pull pod-side fixes (prev_x0 carry, ...)
```

## Baseline launch adjustment

The baseline launch did **not** proceed fully as-is because the runbook/interim ref path was missing on the pod:

```text
/workspace/sources/source_5.png exists=False
```

The readable source image is:

```text
/workspace/IMTalker/assets/source_5.png exists=True shape=(512, 512, 3)
```

The first launch attempt crashed during `cv2.imread`/`cv2.cvtColor` in `live_pipeline.py:137` because the runbook path was missing. No chat/audio driver ran in that attempt.

The successful baseline run used the same command from the previous interim except:

```bash
--ref_path /workspace/IMTalker/assets/source_5.png
```

All other important flags were preserved, including:

```bash
--debug_session
--dump_reply_dir /workspace/IMTalker/dump_reply
--chunk_sec 1.0
```

## Baseline CLI drive

The successful baseline run:

- launched `launch_live_ws.py` in `/workspace/IMTalker` from `/workspace/preprocess_5090/bin/activate`
- drained `/ws/stream` using `tools/ws_capture.py`
- sent `/workspace/IMTalker/assets/audio_3.wav` to `/api/chat` with `sphn.OpusStreamWriter`
- used the chat packet shape `b"\x01" + opus_page`
- killed the server after the capture/driver phase
- ran `/workspace/bin/boundary_summary.py /workspace/IMTalker/live_ws_debug.log`

Driver result:

```text
DRIVER_DONE sent_sec=14.68 opus_pages=367 opus_bytes=59396
```

Note: the pod copy of `audio_3.wav` is only 14.68 seconds, so the source audio was exhausted before a full 30 seconds of mic input. The stream capture window was still longer than that.

## Pulled artifacts

Baseline artifacts were rsynced to local:

```text
logs/baseline_before_boundary_summary.txt
logs/baseline_before_live_ws_debug.log
logs/baseline_before_ws_capture.log
logs/baseline_before_ws_capture/
logs/baseline_before_dump_reply/
```

The raw log was only pulled to disk; it was not pasted into chat.

## Baseline boundary summary output

Record this verbatim as the `BEFORE` baseline in the final report:

```text
log: /workspace/IMTalker/live_ws_debug.log
chunks: 15
boundary_l2:
  mean   : 0.136
  median : 0.139
  p90    : 0.196
  p95    : 0.230
  p99    : 0.268
  max    : 0.278
  spikes (>1.0, top 10 by value): 0 total
mimi (from DBG/mimi, 17 entries):
  full_samples mean=23944 min=23040 max=24960
  remainder    mean=508 min=0 max=960
  empty-chunk events (full_samples=0): 0
x0 head_norm (17 entries):
  mean=4.875 min=4.787 max=5.531
chunk_age (45 entries):
  mean=0.655 p95=0.878 max=0.887
```

## Important observation

This particular CLI baseline did **not** reproduce the 2-7 `boundary_l2` spikes from the 2026-04-19 snapshot; it is already under the acceptance thresholds for boundary stats. Do **not** conclude P1 is unnecessary. The task still requires the overlap accumulator fix because the code-level bug is real: one-second chunks only produce ~12-13 raw Mimi latents, while `overlap_context_latent_count=24`, so the overwrite path can keep `used_overlap_context` disabled.

The next writer should parse `used_overlap_context` from `logs/baseline_before_live_ws_debug.log` before patching or immediately after patching, so the final report has the before/after ratio.

Suggested local parser:

```bash
python3 - <<'PY'
from pathlib import Path
import re
p = Path('logs/baseline_before_live_ws_debug.log')
text = p.read_text(errors='replace')
vals = re.findall(r'used_overlap_context=(True|False)', text)
print(f'used_overlap_context entries={len(vals)} true={vals.count("True")} false={vals.count("False")} ratio={(vals.count("True")/len(vals) if vals else 0):.3f}')
PY
```

## Next exact step

Continue with P1:

1. Inspect `live_pipeline.py` around the existing `_overlap_latents` update in `_render_reply_chunk`.
2. Replace the overwrite:

```python
self._overlap_latents = latents[-self.overlap_context_latent_count:].clone()
```

with the accumulator described in the task:

```python
if self._overlap_latents is None:
    overlap_tail = latents
else:
    overlap_tail = torch.cat([self._overlap_latents, latents], dim=0)
self._overlap_latents = overlap_tail[-self.overlap_context_latent_count:].clone()
```

3. Preserve existing reset points in `__init__`, warmup, and `reset_reply`.
4. Optionally add gated `[DBG/overlap]` with `buffer_size` and `have_enough`.
5. Rsync local -> pod.
6. Rerun the same baseline command, keeping the adjusted ref path:

```bash
--ref_path /workspace/IMTalker/assets/source_5.png
```

7. Run boundary summary again and compare to the before output above.

## Do not forget

- `chunk_sec = 1.0` remains non-negotiable.
- No training has been started or approved.
- No source edits were made in this chat.
- No commit was made in this chat.
