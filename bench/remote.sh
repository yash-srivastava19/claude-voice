#!/bin/sh
# Runs the benchmark on another machine over SSH and copies the results back here.
#
#   sh bench/remote.sh <ssh-host> [cpu|cuda] [ssh options...]
#
# Copies this checkout (no git push needed), runs install.sh there as a user would, adds
# the benchmark's own packages, runs bench/run.py, and brings bench/results/ home. With
# cuda it also swaps in the GPU builds of onnxruntime and sherpa-onnx.
set -eu
HOST=$1
PROVIDER=${2:-cpu}
shift; [ $# -gt 0 ] && shift
SSH="ssh $*"
HERE=$(cd "$(dirname "$0")/.." && pwd)

rsync -az --delete -e "$SSH" --exclude .git --exclude __pycache__ --exclude bench/results "$HERE/" "$HOST:claude-voice/"
$SSH "$HOST" sh -s "$PROVIDER" <<'EOF'
set -eu
PROVIDER=$1
# install.sh checks for the audio tools a user's machine has; a server needs them added.
command -v arecord >/dev/null || { sudo apt-get update -q && sudo apt-get install -y -q curl alsa-utils; } >/dev/null
cd ~/claude-voice
sh install.sh
PY=~/.local/share/voice-mode/venv/bin/python
~/.local/bin/uv pip install -q --python "$PY" soundfile pyarrow jiwer
if [ "$PROVIDER" = cuda ]; then
  ~/.local/bin/uv pip install -q --python "$PY" onnxruntime-gpu
  ~/.local/bin/uv pip install -q --python "$PY" "sherpa-onnx==$("$PY" -c 'import sherpa_onnx; print(sherpa_onnx.__version__)')+cuda" \
    -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
fi
"$PY" helper/voiced.py --fetch-voice ryan && "$PY" helper/voiced.py --fetch-voice lessac
nohup "$PY" bench/run.py all --provider "$PROVIDER" > ~/bench.log 2>&1
EOF
mkdir -p "$HERE/bench/results"
rsync -az -e "$SSH" "$HOST:claude-voice/bench/results/" "$HERE/bench/results/"
echo "results in $HERE/bench/results"
