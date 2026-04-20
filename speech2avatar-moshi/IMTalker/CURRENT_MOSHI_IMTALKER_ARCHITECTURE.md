# Current Moshi + IMTalker Architecture

This file explains the current combined live architecture used by the paired repos in:

```text
/home/user/D/working_yay/both/IMTalker
/home/user/D/working_yay/both/moshi
```

These are source-equivalent to:

```text
/home/user/D/IMTalker
/home/user/D/moshi
```

So this document describes the current working combined system.

## Goal

The goal of this integration is:

- use Moshi as the live conversational engine
- use IMTalker as the speech-driven avatar renderer
- output one synchronized avatar stream to the browser
- avoid playing raw Moshi audio separately in the page

In plain words:

1. user speaks
2. Moshi listens and generates reply text + reply audio
3. IMTalker turns that reply audio into talking-face video
4. the browser plays one muxed audio+video stream

## Main integration files

The core integration does not happen by heavily modifying the original Moshi internals.

It happens mainly in these IMTalker-side wrapper files:

- [launch_live.py](/home/user/D/working_yay/both/IMTalker/launch_live.py)
- [live_pipeline.py](/home/user/D/working_yay/both/IMTalker/live_pipeline.py)

The most relevant supporting model files are:

- [generator/FM.py](/home/user/D/working_yay/both/IMTalker/generator/FM.py)
- [renderer/models.py](/home/user/D/working_yay/both/IMTalker/renderer/models.py)

On the Moshi side, the main referenced pieces are:

- `/home/user/D/working_yay/both/moshi/moshi/moshi/models/lm.py`
- `/home/user/D/working_yay/both/moshi/moshi/moshi/models/loaders.py`
- `/home/user/D/working_yay/both/moshi/client/src/pages/Conversation/hooks/useSocket.ts`

## High-level structure

There are four active layers.

### 1. Visible browser page

Served from `_VIEWER_HTML` inside `launch_live.py`.

This page:

- displays the avatar `<video>`
- fetches `/stream.mp4`
- starts the hidden Moshi iframe at `/moshi`
- polls `/api/stream_state`

### 2. Hidden Moshi browser client

This is the regular Moshi frontend running in an invisible iframe.

It handles:

- microphone capture
- websocket connection to `/api/chat`
- sending Opus audio packets
- receiving text/control messages

It does not play final audio to the user in this combined architecture.

### 3. Moshi live backend

Implemented inside `MoshiAvatarServerState` in `launch_live.py`.

It handles:

- websocket input
- Mimi encoding of user audio
- LM token generation
- text token handling
- reply audio decoding
- forwarding reply PCM to IMTalker

### 4. IMTalker live rendering backend

Implemented in `live_pipeline.py`.

It handles:

- buffering reply PCM
- chunking reply audio
- converting reply PCM into Mimi continuous latents
- calling IMTalker `FMGenerator.sample(...)`
- rendering frames with `IMTRenderer`
- pushing synced audio+video chunks into the fMP4 stream session

## The exact bridge from Moshi to IMTalker

This is the most important integration point.

In [launch_live.py](/home/user/D/working_yay/both/IMTalker/launch_live.py), Moshi is created in `build_moshi_state(...)`.

When `MoshiAvatarServerState` is instantiated, it is wired like this:

- `output_handler=session.handle_moshi_output`
- `user_audio_handler=session.handle_user_audio`
- `send_audio_to_client=False`

That means:

- reply audio from Moshi is handed to IMTalker
- raw Moshi reply audio is not sent to the browser for playback

This is the line where the two systems are joined conceptually:

- Moshi produces reply audio
- IMTalker consumes reply audio

## Detailed live flow

### Step 1. Browser starts

User opens:

```text
http://localhost:8998/
```

The page waits for the user to press Start.

When Start is pressed:

- microphone permission is primed
- `/stream.mp4` fetch begins
- hidden iframe is pointed to `/moshi`

### Step 2. Hidden iframe connects to backend

The hidden Moshi client opens a websocket to:

```text
/api/chat
```

That route is served by `MoshiAvatarServerState.handle_chat(...)`.

### Step 3. User audio goes into Moshi

The hidden client sends websocket binary packets containing microphone audio.

`recv_loop(...)` in `launch_live.py`:

- receives websocket packets
- decodes Opus into PCM
- accumulates enough audio for a Mimi frame
- calls `self.mimi.encode(...)`
- calls `self.lm_gen.step(...)`

This is the live conversational inference loop.

### Step 4. Moshi generates text and reply audio

When `LMGen.step(...)` returns tokens, `decode_and_send(...)` is called.

Inside that function:

- text token is decoded and counted
- audio latents are obtained with `self.mimi.decode_latent(tokens[:, 1:])`
- reply PCM is obtained with `self.mimi.decode(tokens[:, 1:])`

Then the key handoff happens:

- `self.output_handler(tokens, main_pcm, main_latents)`

Since `output_handler` is `session.handle_moshi_output`, that means the reply PCM is now flowing into IMTalker.

### Step 5. IMTalker receives reply PCM

In [live_pipeline.py](/home/user/D/working_yay/both/IMTalker/live_pipeline.py), `handle_moshi_output(...)`:

- flattens reply PCM
- appends it to `_reply_pcm_buffer`
- slices full render chunks of size `chunk_samples`
- pushes those chunks into the render queue

Current chunking behavior is controlled by:

- `chunk_sec`
- `chunk_samples`

This is what converts the raw streaming reply audio into renderable units.

### Step 6. Dedicated render worker processes chunks

The render queue is drained by a dedicated thread, not by an asyncio render task.

This was an important fix.

Why:

- earlier versions let the websocket loop starve the render path
- moving rendering to a thread made live chunk draining stable

The worker loop:

1. pops a queued chunk
2. calls `_render_reply_chunk(...)`
3. calls `_push_to_av(...)`

### Step 7. Reply PCM is converted into IMTalker conditioning

Inside `_render_reply_chunk(...)`, the current compatibility path is:

```text
reply PCM
-> Mimi encode_to_latent(quantize=False)
-> continuous latents
-> align to target frame count
-> FMGenerator.sample(a_feat=...)
-> IMTRenderer.decode(...)
-> RGB frames
```

This is important:

- the system does not directly use `main_latents = self.mimi.decode_latent(...)` as the generator conditioning
- instead it re-encodes the reply PCM through `encode_to_latent(..., quantize=False)`

That choice was made to better match the conditioning distribution used by the IMTalker generator path we were adapting.

### Step 8. IMTalker model internals

The active IMTalker path uses:

- `FMGenerator` from `generator/FM.py`
- `IMTRenderer` from `renderer/models.py`

At a high level:

- `FMGenerator.sample(...)` predicts motion latent trajectories using flow matching
- `IMTRenderer` turns those latents into face frames conditioned on the reference image

Important persistent state:

- `fm_stream_state`

This state is preserved across reply chunks so adjacent chunks animate smoothly rather than restarting from scratch every time.

### Step 9. Audio and frames are realigned

After rendering, `_push_to_av(...)`:

- converts `[N, 3, H, W]` float frames into `[N, H, W, 3]` uint8
- resamples 24 kHz reply PCM to 48 kHz
- trims or pads audio so that:
  - audio duration exactly matches frame duration

This exact duration match is one of the key sync fixes in the current system.

Without this, the browser would drift or desync between audio and lip motion.

### Step 10. fMP4 session paces the stream

`FMP4StreamSession` in `launch_live.py` is the live pacing layer.

It:

- accepts rendered reply chunks
- keeps a play queue
- emits one synchronized tick at a time:
  - one video frame
  - two 20 ms audio packets

When no reply chunk is available, it emits:

- the idle reference frame
- silence

This avoids timeline stalls.

### Step 11. PyAV muxes the final stream

`serve_fmp4_stream(...)`:

- creates the active viewer session
- creates an MP4 muxer via PyAV
- adds:
  - H.264 video stream
  - AAC mono audio stream
- drains the paced ticks into MP4 fragments
- writes them to the HTTP response

The browser therefore receives one live MP4 stream with one shared timeline.

### Step 12. Browser plays the final avatar

The visible page uses:

- `MediaSource`
- `SourceBuffer`
- the `/stream.mp4` fetch stream

So the browser is not trying to manually align separate audio and video tracks.

The MP4 timestamps already enforce sync.

## Why this integration works well

These are the main design decisions that made the combined system workable.

### 1. One final muxed stream

This is the biggest architectural win.

Instead of:

- raw Moshi audio in browser
- separate IMTalker video in browser

we now use:

- one muxed fMP4 stream

That removes a lot of browser-side sync pain.

### 2. `send_audio_to_client=False`

This means the browser does not play reply audio straight from the websocket.

The only audible output is the avatar stream.

This is critical because otherwise users would hear:

- raw Moshi reply
- avatar reply

and the two would drift.

### 3. Dedicated render worker thread

This fixed the old starvation problem where Moshi’s recv loop could monopolize the event loop.

Now:

- Moshi websocket logic can continue
- IMTalker rendering can continue independently

### 4. Audio/video duration matching before mux

This fixed a major source of lip-sync mismatch.

### 5. Reply-state resets

At the beginning of each new websocket turn, IMTalker render state is reset:

- render queue is cleared
- `fm_stream_state` is cleared
- Mimi reply streaming state is reset

But the fMP4 stream itself can stay alive across turns.

That means:

- clean animation state per reply
- no need to restart the browser media element every turn

### 6. Tail-chunk padding for CUDA-graph stability

When renderer/Mimi graph paths are shape-sensitive, a short final tail chunk can break them.

Current code pads short tail chunks to the normal chunk size for rendering while preserving the original valid sample count for duration logic.

That fixed the final short-tail graph error.

## Important runtime options

### IMTalker-side compile

Used through:

```bash
IMTALKER_TORCH_COMPILE=decode_default
```

or:

```bash
IMTALKER_TORCH_COMPILE=1
```

Meaning:

- `decode_default`
  - safer compile path
- `1` / `frame_decoder`
  - CUDA-graph style frame decoder compile path
  - can be faster but more shape-sensitive

### Moshi-side text generation knobs

Current launcher supports:

- `--text_topk`
- `--text_temperature`

These are passed into `LMGen`.

### Chunking

Current live chunking is controlled by:

- `--chunk_sec`

Smaller chunks:

- improve responsiveness
- reduce visible gaps
- increase chunk count and overhead

## Current browser/control architecture

There are actually two browser surfaces:

### Visible page

At:

```text
/
```

It:

- plays the avatar stream
- shows transcript text
- controls Start / Reset

### Hidden page

At:

```text
/moshi
```

It:

- runs the Moshi frontend
- captures mic
- speaks websocket protocol

So the page the user sees is not the same thing as the Moshi app page.

The Moshi app page is used as a hidden controller.

## Debug and observability

The current architecture includes multiple debug layers.

### Server-side Moshi debug

Includes:

- mic chunks received
- reply generated seconds
- generation rate
- token counts
- sentence counts

### Server-side IMTalker debug

Includes:

- render queue length
- chunks enqueued/rendered/pushed
- reply generated seconds
- generation rate
- frame residual

### fMP4 debug

Includes:

- play queue size
- AV queue size
- reply vs idle mode transitions
- dropped ticks

### Browser debug

If opened with:

```text
/?debug=1
```

the browser logs:

- first frame timing
- player events
- current `/api/stream_state`

## The simplest summary

If you want the shortest true description of the current integration, it is this:

```text
Hidden Moshi client sends live mic audio to launch_live.py.
launch_live.py runs Moshi and gets reply PCM + text.
That reply PCM is handed into live_pipeline.py.
live_pipeline.py converts reply PCM into Mimi continuous latents,
then into IMTalker motion latents, then into rendered frames.
Those frames plus the same reply audio are realigned and pushed into
FMP4StreamSession.
serve_fmp4_stream() muxes them into one live MP4 stream.
The visible browser page plays that one stream.
```

That is the current working Moshi + IMTalker architecture.
