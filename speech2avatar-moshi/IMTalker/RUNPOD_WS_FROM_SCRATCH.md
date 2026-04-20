# RunPod From Scratch: Current One-Page WS Demo

This is the shortest practical path to bring up the **current single-page WebSocket demo** for Moshi + IMTalker on a fresh RunPod pod.

This guide assumes:

- pod workspace root is `/workspace`
- final app should run from:
  - `/workspace/IMTalker`
  - `/workspace/moshi`
- you want the **current one-page WS viewer** on port `8998`
- you have your **patched local IMTalker repo** here:
  - `/home/user/D/working_yay/both/IMTalker`

Important:

- Do **not** hardcode your real Hugging Face token into a markdown file.
- Use `export HF_TOKEN=...` at runtime instead.

## 1. Fresh pod bootstrap

Run this on the pod first.

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0))
x = torch.tensor([1.0], device="cuda")
print("tensor:", x)
PY

apt-get update && apt-get install -y python3.11 python3.11-venv ffmpeg git htop tmux && \
cd /workspace && \
git clone https://github.com/bigai-nlco/IMTalker.git && \
git clone https://github.com/kyutai-labs/moshi.git && \
python3.11 -m venv /workspace/preprocess_5090 && \
source /workspace/preprocess_5090/bin/activate && \
python -m pip install --upgrade pip wheel && \
python -m pip install "setuptools==80.9.0" && \
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && \
pip install -r /workspace/IMTalker/requirement.txt && \
pip install "huggingface_hub[cli]" tensorboard && \
pip install hf_transfer && \
pip install "sphn>=0.2.0,<0.3.0" && \
pip install einops sentencepiece aiohttp av
```

## 2. Sync your patched local IMTalker tree to the pod

Run this on your **local machine**, not on the pod.

Replace the host and port if needed.

```bash
export POD_HOST=213.173.111.170
export POD_PORT=24673
export POD_USER=root
export POD_KEY=~/.ssh/id_ed25519

RS="ssh -i $POD_KEY -p $POD_PORT"

rsync -avz --delete --progress \
  --exclude '.git' \
  --exclude '__pycache__' \
  -e "$RS" \
  /home/user/D/working_yay/both/IMTalker/ \
  "$POD_USER@$POD_HOST:/workspace/IMTalker/"
```

Notes:

- This syncs the **current patched one-page WS version**, including:
  - `launch_live.py`
  - `launch_live_ws.py`
  - `live_pipeline.py`
  - `live_pipeline_ws.py`
  - `run_live_avatar_ws.sh`
  - `web_vendor/*`
- You do **not** need to rsync your local `moshi` repo for the current path unless you have your own Moshi changes.

## 3. Download checkpoints

Run this on the pod.

```bash
source /workspace/preprocess_5090/bin/activate
cd /workspace/IMTalker

export HF_TOKEN=YOUR_HF_TOKEN_HERE
huggingface-cli login --token "$HF_TOKEN"

mkdir -p /workspace/IMTalker/checkpoints
mkdir -p /workspace/IMTalker/ckpts_mimi
```

### 3.1 Renderer checkpoint

```bash
huggingface-cli download \
  cbsjtu01/IMTalker \
  renderer.ckpt \
  --local-dir /workspace/IMTalker/checkpoints
```

Expected path:

```text
/workspace/IMTalker/checkpoints/renderer.ckpt
```

### 3.2 Mimi-conditioned generator checkpoint

```bash
huggingface-cli download \
  niloy629/imtalker_mimi_step100000 \
  step=300000.ckpt \
  --local-dir /workspace/IMTalker/ckpts_mimi
```

Expected path:

```text
/workspace/IMTalker/ckpts_mimi/step=300000.ckpt
```

If you already have a newer custom generator checkpoint, use that path instead when launching.

## 4. Quick sanity checks

Run this on the pod.

```bash
source /workspace/preprocess_5090/bin/activate
cd /workspace/IMTalker

python3 -m py_compile launch_live.py launch_live_ws.py live_pipeline.py live_pipeline_ws.py
bash -n run_live_avatar_ws.sh

ls /workspace/IMTalker/checkpoints/renderer.ckpt
ls /workspace/IMTalker/ckpts_mimi/step=300000.ckpt
ls /workspace/IMTalker/assets/source_5.png
ls /workspace/moshi
```

## 5. Launch the current one-page WS demo

Run this on the pod.

```bash
source /workspace/preprocess_5090/bin/activate
cd /workspace/IMTalker

chmod +x run_live_avatar_ws.sh

./run_live_avatar_ws.sh \
  --host 0.0.0.0 \
  --port 8998 \
  --ref_path /workspace/IMTalker/assets/source_5.png \
  --generator_path /workspace/IMTalker/ckpts_mimi/step=300000.ckpt \
  --renderer_path /workspace/IMTalker/checkpoints/renderer.ckpt \
  --hf_repo kyutai/moshiko-pytorch-bf16 \
  --crop \
  --chunk_sec 1.0 \
  --render_batch_size 2 \
  --ws_audio_buffer_ms 80 \
  --debug_session
```

## 6. Open it in the browser

For RunPod proxy, open:

```text
https://YOUR-RUNPOD-ID-8998.proxy.runpod.net/
```

Example shape:

```text
https://nlvrv13ml4j6xy-8998.proxy.runpod.net/
```

Expected behavior:

- one page
- one `Connect` button
- no second visible tab required
- mic prompt appears in the same page
- transcript updates in-page
- avatar A/V plays in-page

## 7. Useful restart loop

If you change code locally:

1. rerun the `rsync` command from section 2
2. restart the launcher on the pod

## 8. If you want the original Gradio app instead

This is the stock IMTalker app, not the current one-page WS demo:

```bash
source /workspace/preprocess_5090/bin/activate
cd /workspace/IMTalker
python app.py
```

Use that only if you want the original repo demo.  
For the current live one-page Moshi + IMTalker setup, use `run_live_avatar_ws.sh`.
