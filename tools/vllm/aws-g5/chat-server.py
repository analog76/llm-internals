#!/usr/bin/env python3
"""Local chat UI server for vLLM.

Serves chat.html at http://127.0.0.1:3000 and proxies /v1/* to the
vLLM server (default http://127.0.0.1:8000) without buffering, so
streamed completions reach the browser as they are generated.

This is a same-origin front-end, which avoids CORS. Bind is localhost
only: the proxy would otherwise expose vLLM on your LAN.

Environment:
  PORT       Listen port (default 3000)
  VLLM_URL   Upstream vLLM base URL (default http://127.0.0.1:8000)
"""

from __future__ import annotations

import http.client
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
CHAT_HTML = HERE / "chat.html"

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = int(os.environ.get("PORT", "3000"))
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")

_UPSTREAM = urlparse(VLLM_URL)
if _UPSTREAM.scheme not in ("http", "https") or not _UPSTREAM.hostname:
    sys.exit(f"VLLM_URL must be an http(s) URL, got: {VLLM_URL!r}")

_UPSTREAM_HOST = _UPSTREAM.hostname
_UPSTREAM_PORT = _UPSTREAM.port or (443 if _UPSTREAM.scheme == "https" else 80)
_UPSTREAM_PATH = (_UPSTREAM.path or "").rstrip("/")  # usually empty
_TIMEOUT = 600  # seconds; first token can take a while after a cold start

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}


def _upstream_conn() -> http.client.HTTPConnection:
    if _UPSTREAM.scheme == "https":
        return http.client.HTTPSConnection(
            _UPSTREAM_HOST, _UPSTREAM_PORT, timeout=_TIMEOUT
        )
    return http.client.HTTPConnection(
        _UPSTREAM_HOST, _UPSTREAM_PORT, timeout=_TIMEOUT
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "vllm-chat-ui/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/chat.html"):
            self._serve_html()
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Connection", "close")
            self.end_headers()
            return
        if path.startswith("/v1/") or path == "/v1":
            self._proxy()
            return
        self._send_bytes(404, b"Not found\n", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path.startswith("/v1/") or path == "/v1":
            self._proxy()
            return
        self._send_bytes(404, b"Not found\n", "text/plain; charset=utf-8")

    def do_OPTIONS(self) -> None:
        # Same-origin only; this is just so a preflight does not 501.
        self.send_response(204)
        self.send_header("Connection", "close")
        self.end_headers()

    def _serve_html(self) -> None:
        if not CHAT_HTML.is_file():
            self._send_bytes(
                500,
                b"chat.html is missing next to chat-server.py\n",
                "text/plain; charset=utf-8",
            )
            return
        body = CHAT_HTML.read_bytes()
        self._send_bytes(200, body, "text/html; charset=utf-8")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        headers = {}
        for key, value in self.headers.items():
            if key.lower() not in HOP_BY_HOP:
                headers[key] = value
        if body is not None:
            headers["Content-Length"] = str(len(body))
        headers["Host"] = (
            f"{_UPSTREAM_HOST}:{_UPSTREAM_PORT}"
            if _UPSTREAM_PORT not in (80, 443)
            else _UPSTREAM_HOST
        )
        headers["Connection"] = "close"
        headers["Accept"] = headers.get("Accept", "*/*")

        upstream_path = _UPSTREAM_PATH + self.path
        conn = _upstream_conn()
        try:
            conn.request(self.command, upstream_path, body=body, headers=headers)
            resp = conn.getresponse()
            self._pipe_response(resp)
        except Exception as exc:
            msg = (
                '{"error":{"message":"Could not reach vLLM at %s (%s). '
                'Is the SSH tunnel up?","type":"proxy_error"}}'
                % (VLLM_URL, type(exc).__name__)
            ).encode()
            try:
                self._send_bytes(502, msg, "application/json")
            except BrokenPipeError:
                pass
        finally:
            conn.close()

    def _pipe_response(self, resp: http.client.HTTPResponse) -> None:
        """Pass the upstream body through in chunks, without buffering it all."""
        self.send_response(resp.status)
        skip = HOP_BY_HOP | {"server", "date"}
        for key, value in resp.getheaders():
            if key.lower() not in skip:
                self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while True:
                # read1 returns as soon as data is available, so SSE
                # chunks are forwarded instead of being coalesced.
                data = resp.read1(8192)
                if not data:
                    break
                self.wfile.write(b"%X\r\n" % len(data))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Browser hit Stop or closed the tab.
            return


def main() -> None:
    if not CHAT_HTML.is_file():
        sys.exit(f"Missing {CHAT_HTML}")
    httpd = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    httpd.allow_reuse_address = True
    print(f"Serving chat UI on http://{LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    print(f"Proxying /v1 -> {VLLM_URL}", flush=True)
    print("Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
