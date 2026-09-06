"""Local dashboard server (standard library only).

Binds to 127.0.0.1 by default: the page carries the Alpaca credentials form, so it
must never be exposed on a network interface without a reverse proxy.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .controller import BotController

STATIC_DIR = Path(__file__).with_name("static")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "TradingBotGUI/1.0"
    controller: BotController  # set on the server instance

    # ------------------------------------------------------------- helpers
    @property
    def ctl(self) -> BotController:
        return self.server.controller  # type: ignore[attr-defined]

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, _json_bytes(payload))

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt, *args):  # quiet the default access log
        return

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            page = (STATIC_DIR / "index.html").read_bytes()
            self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
        elif url.path == "/api/status":
            self._json(self.ctl.snapshot())
        elif url.path == "/api/settings":
            self._json(self.ctl.settings.public())
        elif url.path == "/api/logs":
            since = int((parse_qs(url.query).get("since") or ["0"])[0])
            self._json(self.ctl.logs_since(since))
        elif url.path.startswith("/api/research/"):
            self._research_get(url)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _research_get(self, url) -> None:
        q = parse_qs(url.query)
        first = lambda k, d=None: (q.get(k) or [d])[0]  # noqa: E731
        try:
            if url.path == "/api/research/runs":
                self._json({"runs": self.ctl.research_runs(), "status": {**self.ctl.research_status, "running": self.ctl.research_running}})
            elif url.path == "/api/research/status":
                self._json({**self.ctl.research_status, "running": self.ctl.research_running})
            elif url.path == "/api/research/run":
                self._json(self.ctl.research_summary(first("id", "")))
            elif url.path == "/api/research/trades":
                self._json({"trades": self.ctl.research_trades(first("id", ""), int(first("limit", "0")) or None)})
            elif url.path == "/api/research/trade":
                self._json(self.ctl.research_trade(first("id", ""), int(first("trade", "0"))))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)

    # --------------------------------------------------------------- POST
    def do_POST(self) -> None:
        url = urlparse(self.path)
        try:
            if url.path == "/api/settings":
                self._json(self.ctl.update_settings(self._body()))
            elif url.path == "/api/start":
                data = self._body()
                if data.get("settings"):
                    self.ctl.update_settings(data["settings"])
                self._json(self.ctl.start())
            elif url.path == "/api/stop":
                self._json(self.ctl.stop())
            elif url.path == "/api/test_connection":
                data = self._body()
                if data.get("settings"):
                    self.ctl.update_settings(data["settings"])
                self._json(self.ctl.test_connection())
            elif url.path == "/api/download":
                data = self._body()
                if data.get("settings"):
                    self.ctl.update_settings(data["settings"])
                self._json(self.ctl.download_history())
            elif url.path == "/api/research/start":
                data = self._body()
                if data.get("settings"):
                    self.ctl.update_settings(data["settings"])
                self._json(self.ctl.start_research(data.get("options") or {}))
            elif url.path == "/api/research/stop":
                self._json(self.ctl.stop_research())
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, controller: BotController, host: str = "127.0.0.1", port: int = 8765):
        super().__init__((host, port), DashboardHandler)
        self.controller = controller

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"


def serve(settings_path: str = "settings.json", host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True, block: bool = True) -> DashboardServer:
    controller = BotController(settings_path)
    server = DashboardServer(controller, host, port)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(server.url)).start()
    print(f"dashboard: {server.url}  (settings file: {Path(settings_path).resolve()})")
    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("shutting down")
        finally:
            if controller.running:
                controller.stop()
            server.server_close()
    else:
        threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
    return server


__all__ = ["DashboardServer", "DashboardHandler", "serve"]
