"""Voice daemon for the Claude Code `voice` mod: the entry point. The parts live in
the voice package next to this file; voice/pipeline/task.py wires them together.

Listens on the mic, detects turns with Silero VAD, transcribes with whisper.cpp,
and speaks text with Kokoro. The mod talks to it over HTTP on a Unix socket:

  GET  /events          drain queued events (JSON list); also the watchdog ping
  POST /say  {text,gen} queue a sentence; dropped if gen is older than current
  POST /flush {gen}     stop speaking, drop queued sentences older than gen
  POST /listen {on}     pause or resume the mic
  POST /config {voice,speed}  change the voice or speaking rate (HTTP 400 if unknown)
  POST /shutdown        exit

`voiced.py --voices` prints the voices as JSON; `voiced.py --check-voice NAME`
exits 0 if NAME is a voice. Neither loads a model.

`voiced.py --start` starts the daemon if no live one owns the socket, then exits.
The daemon exits on its own when nothing polls /events for WATCHDOG_S seconds,
so a closed session or a reloaded mod never leaves it holding the mic.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice.config import (BUILD, KOKORO_SPEED, KOKORO_VOICE, LOG, PARAKEET_MODEL, SLEEP_AFTER_S,  # noqa: E402
                          SOCK, STT_ENGINE, VERSION, WAKE_MODEL, WHISPER_MODEL)
from voice.log import log  # noqa: E402
from voice.pipeline.server import UnixHTTPServer, alive, make_handler, request, running_version  # noqa: E402
from voice.power import hold_power, release_power  # noqa: E402
from voice.services.voices import list_voices  # noqa: E402


def start_detached() -> int:
    version = running_version()
    if version == f"{VERSION}+{BUILD}":
        print(json.dumps({"status": "running", "socket": str(SOCK)}))
        return 0
    if version is not None:  # a daemon from before an update: replace it
        log(f"replacing daemon {version or 'of an old version'} with {VERSION}+{BUILD}")
        try:
            request("POST", "/shutdown")
        except OSError:
            pass
        for _ in range(50):
            time.sleep(0.1)
            if not alive():
                break
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "ab") as logf:
        subprocess.Popen(
            [sys.executable, __file__], stdin=subprocess.DEVNULL, stdout=logf, stderr=logf,
            start_new_session=True,
        )
    for _ in range(250):  # Kokoro takes a few seconds to load and warm up
        time.sleep(0.1)
        if alive():
            print(json.dumps({"status": "started", "socket": str(SOCK)}))
            return 0
    print(f"daemon did not come up; see {LOG}", file=sys.stderr)
    return 1


def build():
    """Picks the models and devices and wires them into a VoiceTask."""
    from voice.audio.vad.silero import SileroVAD
    from voice.pipeline.task import VoiceTask, echo_mode
    from voice.services import stt_service, tts_service
    from voice.transports.local.audio import Mic, Player, pick_devices

    tts = tts_service.TTSSwitcher(KOKORO_VOICE, KOKORO_SPEED)
    log("tts:", f"{tts.voice.engine} ({tts.voice.name}, speed {tts.speed})")
    stt = make_stt(stt_service)
    wake_stt = make_wake_stt(stt_service)
    source, sink, aec = pick_devices()
    echo = echo_mode(aec)
    log(f"audio: mic {source or 'default'}, speaker {sink or 'default'}, echo {echo}"
        + (" (mic paused while speaking: no echo cancellation)" if echo == "gate" else ""))
    return VoiceTask(stt=stt, tts=tts, vad=SileroVAD(), mic=Mic(source), player=Player(sink),
                     echo=echo, wake_stt=wake_stt)


def make_stt(stt_service):
    """The configured speech-to-text engine, falling back to whisper if it can't load."""
    if STT_ENGINE != "whisper":
        try:
            stt = stt_service.create(STT_ENGINE, model=PARAKEET_MODEL if STT_ENGINE == "parakeet" else None)
            log(f"stt: {STT_ENGINE} ({PARAKEET_MODEL if STT_ENGINE == 'parakeet' else ''})")
            return stt
        except Exception as err:
            log(f"stt: {STT_ENGINE} unavailable ({err!r}); using whisper")
    # base.en is installed only with VOICE_STT=whisper; tiny.en always is (the wake word uses it).
    model = WHISPER_MODEL if Path(WHISPER_MODEL).exists() else WAKE_MODEL
    log(f"stt: whisper ({model})")
    return stt_service.create("whisper", model=model, language="en")


def make_wake_stt(stt_service):
    """The small model that listens for "hey Claude" while asleep, or None for no auto-sleep."""
    if SLEEP_AFTER_S <= 0 or not Path(WAKE_MODEL).exists():
        log("auto-sleep off")
        return None
    try:
        wake_stt = stt_service.create(
            "whisper", model=WAKE_MODEL, threads=4, short_window=True,
            single_segment=True, no_context=True, max_tokens=10,
            initial_prompt="Hey Claude.",  # nudges a muffled "hey Claude" toward its spelling
        )
    except Exception as err:
        log("wake model unavailable, auto-sleep off:", repr(err))
        return None
    log(f"wake: sleeps after {SLEEP_AFTER_S:.0f}s of quiet; wakes on 'hey Claude'")
    return wake_stt


def main():
    if "--start" in sys.argv:
        sys.exit(start_detached())
    if "--voices" in sys.argv:
        print(json.dumps(list_voices()))
        sys.exit(0)
    if "--check-voice" in sys.argv:  # exit 0 usable, 2 needs --fetch-voice, 1 no such voice
        from voice.services.piper.tts import NotDownloaded
        from voice.services.tts_service import check
        from voice.services.voices import resolve
        name = sys.argv[sys.argv.index("--check-voice") + 1] if len(sys.argv) > sys.argv.index("--check-voice") + 1 else ""
        try:
            check(name)
        except NotDownloaded:
            spec = resolve(name)
            print(f"{name} isn't downloaded yet" + (f"; fetching it from {spec.download}" if spec.download else ""), file=sys.stderr)
            sys.exit(2 if spec.download else 1)
        except (ValueError, OSError) as err:
            print(err, file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if "--fetch-voice" in sys.argv:
        from voice.services.download import fetch_folder
        from voice.services.piper.tts import folder
        from voice.services.voices import resolve
        spec = resolve(sys.argv[sys.argv.index("--fetch-voice") + 1])
        if not spec.download:
            sys.exit(f"{spec.name} has nothing to download")
        fetch_folder(spec.download, folder(spec.ref))
        sys.exit(0)
    if alive():
        log("another daemon owns", SOCK)
        sys.exit(0)
    SOCK.unlink(missing_ok=True)

    hold_power()
    d = build()
    server = UnixHTTPServer(str(SOCK), None)
    server.RequestHandlerClass = make_handler(d)
    for target in d.threads():
        threading.Thread(target=target, daemon=True).start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log("ready on", SOCK)

    d.stopping.wait()
    d.stop()
    release_power()
    server.shutdown()
    SOCK.unlink(missing_ok=True)
    log("stopped")
    os._exit(0)  # skip interpreter teardown; onnxruntime threads abort noisily during it


if __name__ == "__main__":
    main()
