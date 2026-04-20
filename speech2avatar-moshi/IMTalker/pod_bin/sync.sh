#!/usr/bin/env bash
# Prints the rsync commands you need to run LOCALLY to push code to the pod
# and to pull logs back. Copy-paste the command you need.
#
# Requires these env vars set (e.g. in your shell rc):
#   export POD_HOST=root@<runpod-ip>
#   export POD_PORT=<ssh-port>
#   export POD_IMT=/workspace/IMTalker

: "${POD_HOST:?set POD_HOST=root@<ip>}"
: "${POD_PORT:?set POD_PORT=<port>}"
: "${POD_KEY:?set POD_KEY=\$HOME/.ssh/id_ed25519}"
: "${POD_IMT:?set POD_IMT=/workspace/IMTalker}"

LOCAL=/home/user/D/working_yay/both/IMTalker

cat <<EOF
# Push local -> pod (code only, no ckpts, no logs, no media):
rsync -avz --delete \\
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \\
  --exclude 'checkpoints/' --exclude 'exps/' --exclude '*.ckpt' --exclude '*.pt' \\
  --exclude 'logs/' --exclude 'live_ws_debug.log' --exclude 'dump_reply/' \\
  --exclude '*.mp4' --exclude '*.png' --exclude '*.jpg' --exclude '.venv/' \\
  --exclude 'web_vendor/' --exclude 'assets/' \\
  -e "ssh -p \$POD_PORT -i \$POD_KEY" \\
  $LOCAL/ \\
  \$POD_HOST:\$POD_IMT/

# Pull pod -> local logs:
mkdir -p $LOCAL/logs
rsync -avz \\
  -e "ssh -p \$POD_PORT -i \$POD_KEY" \\
  \$POD_HOST:\$POD_IMT/live_ws_debug.log \\
  \$POD_HOST:\$POD_IMT/dump_reply/ \\
  $LOCAL/logs/

# Summarise the latest log (runs on pod, prints tiny summary):
ssh -p \$POD_PORT -i \$POD_KEY \$POD_HOST '/workspace/bin/boundary_summary.py \$POD_IMT/live_ws_debug.log'
EOF
