"""Dashboard tests: settings persistence, controller lifecycle, HTTP API."""

import json
import time
import urllib.request
from pathlib import Path

import pytest

from trading_bot.gui.controller import BotController
from trading_bot.gui.server import DashboardServer
from trading_bot.gui.settings import GuiSettings

FAST_OVERRIDES = "training.window_bars=1400\ntraining.minimum_bars=1400\ntraining.hyperparameter_grid.n_estimators=[30]\n" \
                 "training.acceptance.min_accuracy=0\ntraining.acceptance.min_correlation=-1\ntraining.acceptance.min_net_pnl=-1\n" \
                 "training.acceptance.min_profit_factor=0\ntraining.acceptance.max_drawdown=1\ntraining.acceptance.min_folds_beating_baseline=0\n" \
                 "training.acceptance.require_holdout_edge=false\nmodels.regression.num_threads=1"


def test_settings_roundtrip_keeps_secret_when_blank(tmp_path):
    p = tmp_path / "settings.json"
    s = GuiSettings()
    changed = s.update({"api_key": "PKTEST", "secret_key": "supersecret123", "symbol": "qqq", "fast": "false",
                        "synthetic_bars": "4000", "initial_capital": "50000", "unknown": 1})
    assert set(changed) == {"api_key", "secret_key", "symbol", "fast", "synthetic_bars", "initial_capital"}
    assert s.symbol == "qqq" and s.fast is False and s.synthetic_bars == 4000 and s.initial_capital == 50000.0
    s.save(p)
    loaded = GuiSettings.load(p)
    assert loaded.secret_key == "supersecret123" and loaded.api_key == "PKTEST"
    assert loaded.update({"secret_key": "", "api_key": "PKTEST"}) == []          # blank secret keeps the saved one
    pub = loaded.public()
    assert "secret_key" not in pub and pub["secret_key_set"] and pub["secret_key_hint"].endswith("t123")
    assert GuiSettings.load(tmp_path / "missing.json") == GuiSettings()


def test_controller_backtest_lifecycle(tmp_path):
    ctl = BotController(tmp_path / "settings.json")
    ctl.update_settings({"symbol": "SYN", "mode": "backtest", "data_source": "synthetic", "synthetic_bars": 1500,
                         "synthetic_seed": 3, "fast": True, "artifacts_dir": str(tmp_path / "art"),
                         "overrides": FAST_OVERRIDES})
    cfg = ctl.build_config()
    assert cfg.market.instrument == "SYN" and cfg.training.window_bars == 1400 and cfg.training.acceptance.min_accuracy == 0
    ctl.start()
    with pytest.raises(RuntimeError):
        ctl.start()
    deadline = time.time() + 240
    while ctl.running and time.time() < deadline:
        time.sleep(0.5)
    assert not ctl.running and ctl.phase == "finished", (ctl.phase, ctl.error)
    snap = ctl.snapshot()
    assert snap["bot"]["bars_stored"] == 1500 and snap["bot"]["retrains"] >= 1
    assert snap["bot"]["summary"] is not None and "net_return" in snap["bot"]["summary"]
    assert isinstance(snap["bot"]["equity_curve"], list) and len(snap["bot"]["equity_curve"]) <= 602
    logs = ctl.logs_since(0)
    assert any("finished" in l["text"] for l in logs["lines"])
    assert Path(tmp_path / "art" / "audit").exists()


def test_controller_stop_interrupts_backtest(tmp_path):
    ctl = BotController(tmp_path / "settings.json")
    ctl.update_settings({"symbol": "SYN", "mode": "backtest", "data_source": "synthetic", "synthetic_bars": 6000,
                         "fast": True, "artifacts_dir": str(tmp_path / "art"), "overrides": FAST_OVERRIDES})
    ctl.start()
    time.sleep(1.0)
    ctl.stop()
    deadline = time.time() + 120
    while ctl.running and time.time() < deadline:
        time.sleep(0.2)
    assert not ctl.running and ctl.phase == "finished"
    assert ctl.snapshot()["bot"]["bars_stored"] < 6000
    assert any("interrupted" in l["text"] for l in ctl.logs_since(0)["lines"])


def test_controller_refuses_paper_without_credentials(tmp_path):
    ctl = BotController(tmp_path / "settings.json")
    ctl.update_settings({"mode": "paper"})
    with pytest.raises(RuntimeError):
        ctl.start()
    assert ctl.test_connection()["ok"] is False
    assert ctl.download_history()["ok"] is False


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_http_api(tmp_path):
    import threading
    ctl = BotController(tmp_path / "settings.json")
    server = DashboardServer(ctl, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = server.url.rstrip("/")
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            html = r.read().decode()
        assert "Fractional-Memory Trading Bot" in html and "api_key" in html
        status, s = _post(base + "/api/settings", {"api_key": "PKX", "secret_key": "abcdefgh1234", "symbol": "SYN"})
        assert status == 200 and s["settings"]["api_key"] == "PKX" and s["settings"]["secret_key_set"]
        assert "abcdefgh1234" not in json.dumps(_get(base + "/api/settings")[1])   # the secret never leaves the server
        assert not (tmp_path / "settings.json").read_text().count("PKX") == 0
        status, st = _get(base + "/api/status")
        assert status == 200 and st["phase"] == "idle" and st["bot"] is None
        status, r = _post(base + "/api/stop", {})
        assert status == 200 and r["phase"] == "idle"
        status, r = _post(base + "/api/start", {"settings": {"mode": "paper", "secret_key": ""}})
        assert status == 200, r                                # PKX/secret are saved, so paper mode is allowed to start
        time.sleep(1.5)
        status, st = _get(base + "/api/status")
        assert st["phase"] in ("error", "starting", "running", "finished")
        deadline = time.time() + 60
        while ctl.running and time.time() < deadline:
            time.sleep(0.5)
        assert ctl.phase == "error" and ctl.error           # bogus credentials fail against Alpaca, never crash the server
        status, r = _get(base + "/api/logs?since=0")
        assert status == 200 and r["lines"]
        assert _get(base + "/nope")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
