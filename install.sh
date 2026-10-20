#!/bin/sh
# Installs the voice daemon's Python environment and models (about 500 MB) into
# ~/.local/share/voice-mode. The plugin runs this on the first /talk start; running
# it by hand is fine too. Safe to run again: finished downloads are skipped.
set -eu

V="${VOICE_HOME:-$HOME/.local/share/voice-mode}"
# Bump when a release needs new packages or models; an older install then reruns
# this on the next /talk start (finished downloads are skipped).
SETUP=2
if [ "${1:-}" = "--check" ]; then [ "$(cat "$V/.installed" 2>/dev/null)" = "$SETUP" ]; exit; fi
die() { echo "voice: $*" >&2; exit 1; }

# Audio tools: sox on macOS, ALSA's arecord/aplay on Linux.
case "$(uname -s)" in
  Darwin) command -v rec >/dev/null || die "sox is missing. Run: brew install sox" ;;
  Linux)  command -v arecord >/dev/null && command -v aplay >/dev/null ||
            die "arecord/aplay are missing. Install your distro's alsa-utils package." ;;
  *) die "only macOS and Linux are supported" ;;
esac

PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

echo "Creating Python environment in $V/venv..."
mkdir -p "$V/models"
[ -x "$V/venv/bin/python" ] || uv venv -q "$V/venv" --python 3.12
uv pip install -q --python "$V/venv/bin/python" kokoro-onnx onnxruntime numpy pywhispercpp

echo "Downloading models (about 500 MB, once)..."
get() {  # get <file> <url>
  [ -s "$V/models/$1" ] && return
  echo "  $1"
  curl -fsSL -o "$V/models/$1.part" "$2" && mv "$V/models/$1.part" "$V/models/$1"
}
K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
W=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
get kokoro-v1.0.onnx  "$K/kokoro-v1.0.onnx"
get voices-v1.0.bin   "$K/voices-v1.0.bin"
get ggml-base.en.bin  "$W/ggml-base.en.bin"
get ggml-tiny.en.bin  "$W/ggml-tiny.en.bin"
get silero_vad.onnx   https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

mkdir -p "$V/voices"
echo "$SETUP" > "$V/.installed"
echo "Voice is installed."
