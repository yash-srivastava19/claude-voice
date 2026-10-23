#!/bin/sh
# Installs the voice daemon's Python environment and models (about 500 MB) into
# ~/.local/share/voice-mode. The plugin runs this on the first /talk start; running
# it by hand is fine too. Safe to run again: whatever is already in place is kept,
# and an interrupted download resumes where it stopped.
#
# What it needs, and what it does when something is missing:
#   audio tools  sox on macOS, alsa-utils on Linux: stops and says what to install
#   curl         stops and says so
#   uv           installs it into ~/.local/bin
#   Python       uses one already on the machine (3.11 to 3.13); uv downloads 3.13 only
#                when there is none
# Every step's output goes to install.log; on failure the reason and the last lines
# of that log go to stderr.
set -eu

V="${VOICE_HOME:-$HOME/.local/share/voice-mode}"
# Bump when a release needs new packages or models; an older install then reruns
# this on the next /talk start (finished downloads are skipped).
SETUP=2
# --check: installed enough to run (the marker). --complete: has everything this version
# would install, so an older install can catch up with --background without making
# anyone wait at /talk start.
if [ "${1:-}" = "--check" ]; then [ "$(cat "$V/.installed" 2>/dev/null)" = "$SETUP" ]; exit; fi
if [ "${1:-}" = "--complete" ]; then
  [ -d "$V/models/parakeet-tdt-0.6b-v2-int8" ] && "$V/venv/bin/python" -c "import sherpa_onnx" 2>/dev/null; exit
fi
if [ "${1:-}" = "--background" ]; then
  mkdir "$V/.upgrading" 2>/dev/null || exit 0  # one at a time
  (trap 'rmdir "$V/.upgrading"' EXIT; nohup sh "$0" >> "$V/install.log" 2>&1) </dev/null >/dev/null 2>&1 &
  exit 0
fi

mkdir -p "$V/models" "$V/voices"
LOG="$V/install.log"
: > "$LOG"
die() { echo "voice: $*" >&2; exit 1; }
step() { echo "$*"; echo "== $*" >> "$LOG"; }
# run <what failed> <command...>: the command's output goes to the log only.
run() {
  what=$1; shift
  "$@" >> "$LOG" 2>&1 && return
  { echo "voice: $what. Last lines of $LOG:"; tail -n 8 "$LOG"; } >&2
  exit 1
}

# The dependencies' own bounds: kokoro-onnx wants >=3.10,<3.14, onnxruntime >=3.11.
PYTHONS='>=3.11,<3.14'
# uv drops a request after 30 s by default, which a 70 MB interpreter or a large wheel
# loses to on a slow link.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"

case "$(uname -s)" in
  Darwin) command -v rec >/dev/null && command -v play >/dev/null ||
            die "sox is missing. Run: brew install sox" ;;
  Linux)  command -v arecord >/dev/null && command -v aplay >/dev/null ||
            die "arecord/aplay are missing. Install your distro's alsa-utils package." ;;
  *) die "only macOS and Linux are supported" ;;
esac
command -v curl >/dev/null || die "curl is missing; install it and run /talk start again."

PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
  step "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh -o "$V/uv-install.sh" >> "$LOG" 2>&1 ||
    die "could not download the uv installer (offline?). Install uv yourself: https://docs.astral.sh/uv/"
  run "installing uv failed" sh "$V/uv-install.sh"
  rm -f "$V/uv-install.sh"
  command -v uv >/dev/null || die "uv was installed but is not on PATH; open a new terminal and try again."
fi

# Keep a working environment; rebuild one whose interpreter has gone (a Homebrew
# upgrade removes the old Python) or falls outside the supported versions.
VPY="$V/venv/bin/python"
if [ -e "$V/venv" ] && ! "$VPY" -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] < (3, 14))' 2>/dev/null; then
  step "Rebuilding the Python environment (its interpreter is missing or unsupported)..."
  rm -rf "$V/venv"
fi
if [ ! -x "$VPY" ]; then
  step "Creating Python environment in $V/venv..."
  # A range, not a pin: uv takes a matching interpreter already on the machine and
  # only downloads one when none matches.
  run "could not set up Python 3.11-3.13. With none installed, uv downloads one; on a slow link install it yourself (macOS: brew install python@3.13) and run /talk start again" uv venv -q "$V/venv" --python "$PYTHONS"
fi
step "Installing Python packages..."
run "installing the Python packages failed" \
  uv pip install -q --python "$VPY" kokoro-onnx onnxruntime numpy pywhispercpp sherpa-onnx

step "Downloading models (about 1 GB, once)..."
get() {  # get <file> <url>
  [ -s "$V/models/$1" ] && return
  step "  $1"
  part="$V/models/$1.part"
  # -C - resumes a part left by an interrupted run, so a slow link gets there over
  # several tries. Only a part the server refuses to resume (curl exit 33, or 22 for
  # HTTP 416) is thrown away and fetched again whole.
  rc=0
  curl -fsSL --retry 5 --retry-all-errors -C - -o "$part" "$2" >> "$LOG" 2>&1 || rc=$?
  if [ "$rc" = 33 ] || { [ "$rc" = 22 ] && [ -s "$part" ]; }; then
    rm -f "$part"; rc=0
    curl -fsSL --retry 5 --retry-all-errors -o "$part" "$2" >> "$LOG" 2>&1 || rc=$?
  fi
  [ "$rc" = 0 ] || die "downloading $1 failed (curl exit $rc; offline or a slow link?). Run /talk start again: it resumes. URL: $2"
  mv "$part" "$V/models/$1"
}
K=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
W=https://huggingface.co/ggerganov/whisper.cpp/resolve/main
get kokoro-v1.0.onnx  "$K/kokoro-v1.0.onnx"
get voices-v1.0.bin   "$K/voices-v1.0.bin"
[ "${VOICE_STT:-}" = whisper ] && get ggml-base.en.bin "$W/ggml-base.en.bin"
get ggml-tiny.en.bin  "$W/ggml-tiny.en.bin"
get silero_vad.onnx   https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

unpack() {  # unpack <folder under models/> <url of a .tar.bz2 holding one folder>
  [ -d "$V/models/$1" ] && return
  name=$(basename "$2")
  get "$name" "$2"
  rm -rf "$V/models/.unpack" && mkdir -p "$V/models/.unpack" "$(dirname "$V/models/$1")"
  tar -xjf "$V/models/$name" -C "$V/models/.unpack" || die "unpacking $name failed; delete $V/models/$name and run /talk start again"
  mv "$V/models/.unpack/"* "$V/models/$1" && rm -rf "$V/models/.unpack" "$V/models/$name"
}
R=https://github.com/k2-fsa/sherpa-onnx/releases/download
unpack parakeet-tdt-0.6b-v2-int8 "$R/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2"
# Piper voices (ryan, lessac) are fetched the first time one is picked: voiced.py --fetch-voice.

echo "$SETUP" > "$V/.installed"
echo "Voice is installed."
