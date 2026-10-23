"""The harness-facing API: HTTP over a Unix socket (endpoints in voiced.py's docstring)."""

import json
import os
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from voice.config import SOCK, VERSION
from voice.log import log


class UnixHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True

    def server_bind(self):
        self.socket.bind(self.server_address)
        self.server_name, self.server_port = "voice", 0


def make_handler(d):
    """d: the VoiceTask the endpoints drive."""
    class Handler(BaseHTTPRequestHandler):
        def address_string(self):
            return "unix"

        def log_message(self, *args):
            pass

        def reply(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:  # the health check hangs up once it has read enough
                pass

        def do_GET(self):
            if self.path == "/events":
                self.reply({"events": d.drain(), "speaking": d.out.speaking,
                            "listening": d.listening, "gen": d.out.gen, "awake": d.awake,
                            "echo": d.echo})
            elif self.path == "/health":
                self.reply({"ok": True, "pid": os.getpid(), "version": VERSION, "echo": d.echo})
            elif self.path == "/config":
                self.reply({"voice": d.out.tts.voice.name, "speed": d.out.tts.speed})
            else:
                self.send_error(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/say":
                d.out.say(str(body.get("text", "")), int(body.get("gen", 0)))
            elif self.path == "/config":
                try:
                    d.out.tts.configure(body.get("voice"), body.get("speed"))
                except (ValueError, OSError) as err:
                    self.reply({"error": str(err)}, 400)
                    return
            elif self.path == "/flush":
                d.out.flush(int(body.get("gen", d.out.gen + 1)), str(body.get("why", "mod")))
            elif self.path == "/wake":
                d.wake(str(body.get("why", "mod")))
            elif self.path == "/sleep":
                d.sleep(str(body.get("why", "mod")))
            elif self.path == "/log":
                log("mod:", str(body.get("line", ""))[:300])
            elif self.path == "/listen":
                d.listening = bool(body.get("on", True))
            elif self.path == "/shutdown":
                d.stopping.set()
            else:
                self.send_error(404)
                return
            self.reply({"ok": True, "gen": d.out.gen})

    return Handler


def request(method, path) -> bytes:
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(1)
        s.connect(str(SOCK))
        s.sendall(f"{method} {path} HTTP/1.0\r\nContent-Length: 0\r\n\r\n".encode())
        return s.recv(4096)


def running_version():
    """The live daemon's version ("" for one too old to say), or None if none is live."""
    try:
        reply = request("GET", "/health")
    except OSError:
        return None
    if b'"ok": true' not in reply:
        return None
    m = re.search(rb'"version": "([^"]*)"', reply)
    return m.group(1).decode() if m else ""


def alive() -> bool:
    return running_version() is not None
