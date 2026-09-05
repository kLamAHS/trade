"""BotController: runs the TradingBot in a background thread for the dashboard.

Responsibilities: build the configuration from the saved settings, start / stop a
backtest or the Alpaca paper loop, capture logs and structured events, expose a
JSON-serialisable status snapshot, test the Alpaca connection and download history.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import DEFAULT_CONFIG_PATH, FrozenConfig, load_config
from .settings import GuiSettings


class BotController:
    def __init__(self, settings_path: str | Path = "settings.json", log_capacity: int = 3000):
        self.settings_path = Path(settings_path)
        self.settings = GuiSettings.load(self.settings_path)
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._logs: deque[tuple[int, str, str]] = deque(maxlen=log_capacity)
        self._events: deque[dict] = deque(maxlen=200)
        self._seq = 0
        self.bot = None
        self.phase = "idle"              # idle | starting | running | stopping | finished | error
        self.message = ""
        self.error: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.summary: Optional[dict] = None
        self.run_mode: Optional[str] = None
        self.download_status: Optional[dict] = None
        self._download_thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- logs
    def log(self, message: str, level: str = "info") -> None:
        with self._lock:
            self._seq += 1
            self._logs.append((self._seq, datetime.now(timezone.utc).strftime("%H:%M:%S"), f"{message}"))

    def logs_since(self, since: int = 0, limit: int = 500) -> dict[str, Any]:
        with self._lock:
            lines = [{"seq": s, "time": t, "text": m} for s, t, m in self._logs if s > since][-limit:]
            return {"lines": lines, "seq": self._seq}

    def _on_event(self, record: dict) -> None:
        with self._lock:
            self._events.append(record)

    # ------------------------------------------------------------ settings
    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            changed = self.settings.update(data)
            self.settings.save(self.settings_path)
        return {"changed": changed, "settings": self.settings.public()}

    def build_config(self) -> FrozenConfig:
        s = self.settings
        cfg = load_config(s.config_path or None)
        if s.fast:
            import yaml
            with open(Path(DEFAULT_CONFIG_PATH).with_name("strategy_fast.yaml"), "r", encoding="utf-8") as fh:
                cfg = cfg.with_overrides(yaml.safe_load(fh) or {})
        cfg = cfg.with_overrides({"market": {"instrument": s.symbol.upper()},
                                  "portfolio": {"initial_capital": float(s.initial_capital)},
                                  "alpaca": {"paper": bool(s.paper), "mirror_orders": bool(s.mirror_orders),
                                             "history_days": int(s.history_days)}})
        ov = s.override_dict()
        if ov:
            cfg = cfg.with_overrides(ov)
        return cfg

    # ------------------------------------------------------------ control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise RuntimeError("the bot is already running")
            if self.settings.mode == "paper" and not self.settings.has_credentials():
                raise RuntimeError("paper mode needs an Alpaca API key and secret (Settings)")
            if self.settings.mode == "backtest" and self.settings.data_source == "csv" and not Path(self.settings.csv_path).exists():
                raise RuntimeError(f"CSV file not found: {self.settings.csv_path!r}")
            self._stop.clear()
            self.phase, self.message, self.error, self.summary = "starting", "", None, None
            self.started_at, self.finished_at = datetime.now(timezone.utc).isoformat(), None
            self.run_mode = self.settings.mode
            self._events.clear()
            self._thread = threading.Thread(target=self._run, name="trading-bot", daemon=True)
            self._thread.start()
        return {"phase": self.phase}

    def stop(self) -> dict[str, Any]:
        if not self.running:
            return {"phase": self.phase}
        self._stop.set()
        self.phase = "stopping"
        self.log("stop requested")
        return {"phase": self.phase}

    def _wait(self, seconds: float) -> None:
        self._stop.wait(seconds)

    # ---------------------------------------------------------------- run
    def _run(self) -> None:
        from ..bot import TradingBot

        try:
            cfg = self.build_config()
            s = self.settings
            s.apply_environment()
            run_id = datetime.now(timezone.utc).strftime(f"{s.mode}_%Y%m%dT%H%M%SZ")
            broker = None
            if s.mode == "paper" and s.mirror_orders:
                from ..execution.simulator import AlpacaPaperBroker
                broker = AlpacaPaperBroker(s.api_key, s.secret_key, paper=bool(s.paper))
            bot = TradingBot(cfg, run_id=run_id, artifacts_dir=s.artifacts_dir, broker=broker, log=self.log,
                             async_retrain=(s.mode == "paper"), on_event=self._on_event)
            self.bot = bot
            self.log(f"run {run_id}: {s.mode} on {cfg.market.instrument}, config digest {cfg.digest()}")
            self.phase = "running"
            if s.mode == "backtest":
                self._run_backtest(bot, cfg)
            else:
                self._run_paper(bot, cfg)
            self.phase = "finished"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(f"ERROR {self.error}")
            self.log(traceback.format_exc().strip().splitlines()[-1])
            self.phase = "error"
            try:
                if self.bot is not None and self.summary is None:
                    self.summary = self.bot.finalize()
            except Exception:  # pragma: no cover
                pass
        finally:
            self.finished_at = datetime.now(timezone.utc).isoformat()

    def _run_backtest(self, bot, cfg) -> None:
        from ..data.feed import ReplayFeed
        from ..data.store import read_bars_csv
        from ..data.synthetic import generate_synthetic_bars

        s = self.settings
        if s.data_source == "csv":
            bars = read_bars_csv(s.csv_path, cfg.market.instrument, int(cfg.market.bar_minutes))
        else:
            bars = generate_synthetic_bars(int(s.synthetic_bars), seed=int(s.synthetic_seed),
                                           instrument=cfg.market.instrument, calendar=bot.calendar)
        feed = ReplayFeed(bars, bot.calendar)
        self.message = f"backtest over {len(feed)} bars"
        self.log(self.message)
        t0 = time.time()
        for bar in feed:
            if self._stop.is_set():
                self.log("backtest interrupted")
                break
            bot.on_bar(bar)
        self.summary = bot.finalize(elapsed=time.time() - t0)
        m = self.summary["metrics"]
        self.log(f"finished: net return {m['net_return']:+.4%}, sharpe {m['sharpe']:.2f}, trades {m['trade_count']}")

    def _run_paper(self, bot, cfg) -> None:
        from ..data.feed import AlpacaBarFeed

        s = self.settings
        feed = AlpacaBarFeed(cfg.market.instrument, bot.calendar, api_key=s.api_key, secret_key=s.secret_key,
                             feed=cfg.alpaca.feed, bar_minutes=int(cfg.market.bar_minutes),
                             poll_seconds=int(cfg.alpaca.poll_seconds),
                             adjustment=str(cfg.alpaca.get("adjustment", "split")))
        now = datetime.now(timezone.utc)
        self.message = "downloading history from Alpaca"
        history = feed.fetch_history(now - timedelta(days=int(cfg.alpaca.history_days)), now)
        self.log(f"bootstrapping with {len(history)} completed historical bars")
        self.message = "bootstrapping and training"
        bot.bootstrap(history)
        if len(bot.store):
            feed.seed_last_timestamp(bot.store.last().timestamp)
        self.message = f"live: polling Alpaca every {cfg.alpaca.poll_seconds}s"
        self.log(f"state={bot.state.value} model={bot.registry.current_version}; entering live loop")
        from ..main import run_live_loop

        try:
            run_live_loop(bot, feed, int(cfg.alpaca.poll_seconds), log=self.log,
                          should_stop=self._stop.is_set, sleep=self._wait)
        finally:
            self.summary = bot.finalize()

    # ------------------------------------------------------------- alpaca
    def test_connection(self) -> dict[str, Any]:
        s = self.settings
        if not s.has_credentials():
            return {"ok": False, "error": "enter an API key and secret first"}
        try:
            from alpaca.trading.client import TradingClient

            client = TradingClient(s.api_key, s.secret_key, paper=bool(s.paper))
            acct = client.get_account()
            return {"ok": True, "account": str(getattr(acct, "account_number", "")), "status": str(getattr(acct, "status", "")),
                    "equity": float(getattr(acct, "equity", 0) or 0), "cash": float(getattr(acct, "cash", 0) or 0),
                    "buying_power": float(getattr(acct, "buying_power", 0) or 0), "paper": bool(s.paper)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def download_history(self) -> dict[str, Any]:
        if self._download_thread is not None and self._download_thread.is_alive():
            return {"ok": False, "error": "a download is already running"}
        if not self.settings.has_credentials():
            return {"ok": False, "error": "enter an API key and secret first"}
        self.download_status = {"phase": "running", "message": "downloading"}
        self._download_thread = threading.Thread(target=self._download, name="download", daemon=True)
        self._download_thread.start()
        return {"ok": True}

    def _download(self) -> None:
        try:
            from ..data.calendar import SessionCalendar
            from ..data.feed import AlpacaBarFeed
            from ..data.store import BarStore

            s = self.settings
            cfg = self.build_config()
            cal = SessionCalendar.from_config(cfg)
            feed = AlpacaBarFeed(cfg.market.instrument, cal, api_key=s.api_key, secret_key=s.secret_key,
                                 feed=cfg.alpaca.feed, bar_minutes=int(cfg.market.bar_minutes),
                                 adjustment=str(cfg.alpaca.get("adjustment", "split")))
            now = datetime.now(timezone.utc)
            bars = feed.fetch_history(now - timedelta(days=int(s.history_days)), now)
            store = BarStore(cfg.market.instrument, int(cfg.market.bar_minutes), bars)
            out = Path(s.artifacts_dir) / "data" / f"{cfg.market.instrument}_{cfg.market.bar_minutes}m.csv"
            store.save(out)
            with self._lock:
                self.settings.csv_path = str(out)
                self.settings.data_source = "csv"
                self.settings.save(self.settings_path)
            self.download_status = {"phase": "finished", "message": f"saved {len(store)} bars to {out}", "path": str(out),
                                    "bars": len(store)}
            self.log(self.download_status["message"])
        except Exception as exc:
            self.download_status = {"phase": "error", "message": f"{type(exc).__name__}: {exc}"}
            self.log(f"download failed: {exc}")

    # ------------------------------------------------------------- status
    def snapshot(self) -> dict[str, Any]:
        bot = self.bot
        out: dict[str, Any] = {
            "phase": self.phase, "message": self.message, "error": self.error, "mode": self.run_mode,
            "started_at": self.started_at, "finished_at": self.finished_at, "running": self.running,
            "download": self.download_status, "settings": self.settings.public(),
            "now": datetime.now(timezone.utc).isoformat(),
        }
        if bot is None:
            out["bot"] = None
            return out
        led = bot.ledger
        eq_hist = led.equity_history
        step = max(1, math.ceil(len(eq_hist) / 600))
        curve = [[t.isoformat(), round(e, 2), round(x, 4)] for t, e, x in eq_hist[::step]]
        if eq_hist and (len(eq_hist) - 1) % step:
            t, e, x = eq_hist[-1]
            curve.append([t.isoformat(), round(e, 2), round(x, 4)])
        last_bar = bot.store.last() if len(bot.store) else None
        with self._lock:
            events = list(self._events)[-60:]
        out["bot"] = {
            "run_id": bot.run_id, "state": bot.state.value, "instrument": bot.instrument,
            "bars_processed": bot.processed, "bars_stored": len(bot.store), "rejected_bars": bot.rejected_bars,
            "halted_bars": bot.halted_bars, "retrains": bot.retrain_count, "model_version": bot.registry.current_version,
            "adaptive_d": bot.feature_engine.adaptive_d if bot.model is not None else None,
            "bars_since_retrain": bot.bars_since_retrain, "retrain_every": bot.retrain_every,
            "minimum_bars": bot.minimum_bars,
            "equity": led.equity, "cash": led.cash, "units": led.units, "exposure": led.exposure,
            "drawdown": led.drawdown, "daily_return": led.daily_return, "realized_pnl": led.realized_pnl,
            "unrealized_pnl": led.unrealized_pnl, "equity_peak": led.equity_peak, "initial_capital": led.initial_capital,
            "total_costs": led.total_costs, "n_fills": len(led.fills), "n_trades": len(led.trades),
            "holding_bars": led.holding_bars, "entry_price": led.entry_price,
            "last_bar": {"timestamp": last_bar.timestamp.isoformat(), "close": last_bar.close,
                         "open": last_bar.open, "high": last_bar.high, "low": last_bar.low} if last_bar else None,
            "risk": {"drawdown_halted": bot.risk.drawdown_halted, "ablation_halted": bot.risk.ablation_halted,
                     "data_halted": bot.risk.data_halted, "daily_halt": bot.risk.daily_halt_date is not None,
                     "ablation_failures": bot.risk.ablation_failures},
            "last_decision": bot.last_record,
            "equity_curve": curve,
            "trades": [t.to_dict() for t in led.trades[-50:]],
            "fractional_contribution": bot.diagnostics.contribution[-40:],
            "events": events,
            "summary": self.summary["metrics"] if self.summary else None,
        }
        return out


__all__ = ["BotController"]
