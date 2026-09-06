"""Command-line entry point.

    python -m trading_bot.main backtest --synthetic 6000 --fast
    python -m trading_bot.main backtest --csv bars.csv
    python -m trading_bot.main download --symbol SPY --days 1200 --out artifacts/data/SPY_30m.csv
    python -m trading_bot.main paper --symbol SPY
    python -m trading_bot.main train --csv bars.csv
    python -m trading_bot.main diagnose --csv bars.csv
    python -m trading_bot.main gui                   # browser dashboard (or start.bat / start.sh)

Always run as a module (``python -m trading_bot.main``) or through the
``trading-bot`` console script so that the ``trading_bot.logging`` package
never shadows the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from .config import DEFAULT_CONFIG_PATH, FrozenConfig, load_config


def parse_scalar(text: str):
    """Numbers first (PyYAML 1.1 reads '1e-3' as a string), then YAML for bools/lists/strings."""
    t = text.strip()
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return yaml.safe_load(t)


def _parse_overrides(items: list[str] | None) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_scalar(value)
    return out


def build_config(args) -> FrozenConfig:
    cfg = load_config(args.config)
    if getattr(args, "fast", False):
        with open(Path(DEFAULT_CONFIG_PATH).with_name("strategy_fast.yaml"), "r", encoding="utf-8") as fh:
            cfg = cfg.with_overrides(yaml.safe_load(fh) or {})
    if getattr(args, "symbol", None):
        cfg = cfg.with_overrides({"market": {"instrument": args.symbol}})
    overrides = _parse_overrides(getattr(args, "set", None))
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg


def _load_bars(args, cfg):
    """Raw bars in file order; ordering/integrity is left to the DataValidator (spec section 35)."""
    from .data.calendar import SessionCalendar
    from .data.store import read_bars_csv
    from .data.synthetic import generate_synthetic_bars

    cal = SessionCalendar.from_config(cfg)
    if getattr(args, "synthetic", None):
        return generate_synthetic_bars(int(args.synthetic), seed=int(args.seed), instrument=cfg.market.instrument,
                                       calendar=cal, memory_d=float(args.memory_d), amplitude=float(args.amplitude))
    if getattr(args, "csv", None):
        return read_bars_csv(args.csv, cfg.market.instrument, int(cfg.market.bar_minutes))
    raise SystemExit("provide --csv PATH or --synthetic N")


def _validated_store(cfg, bars):
    """Run bars through the validator and keep the storable ones (for train/diagnose)."""
    from .data.store import BarStore
    from .data.validator import DataValidator

    validator = DataValidator.from_config(cfg)
    store = BarStore(cfg.market.instrument, int(cfg.market.bar_minutes))
    rejected = 0
    for b in bars:
        if validator.validate(b).storable:
            store.append(b)
        else:
            rejected += 1
    if rejected:
        print(f"validator rejected {rejected} bar(s)")
    return store


def cmd_backtest(args) -> int:
    from .bot import TradingBot
    from .data.feed import ReplayFeed

    cfg = build_config(args)
    bars = _load_bars(args, cfg)
    bot = TradingBot(cfg, run_id=args.run_id, artifacts_dir=args.artifacts, log=print if not args.quiet else None)
    feed = ReplayFeed(bars, bot.calendar)
    print(f"backtest: {len(feed)} bars of {cfg.market.instrument}, config digest {cfg.digest()}")
    summary = bot.run(feed, max_bars=args.max_bars)
    _print_summary(summary)
    if getattr(args, "require_model", False) and summary["model_version"] == "none":
        print("no model was accepted during the run (--require-model)")
        return 2
    return 0


def _print_summary(summary: dict) -> None:
    m = summary["metrics"]
    print(json.dumps({
        "run_id": summary["run_id"], "bars": summary["bars_stored"], "final_state": summary["final_state"],
        "retrains": summary["retrains"], "model": summary["model_version"],
        "net_return": round(m["net_return"], 5), "gross_return": round(m["gross_return"], 5),
        "costs": round(m["transaction_costs"], 2), "sharpe": round(m["sharpe"], 3), "sortino": round(m["sortino"], 3),
        "max_drawdown": round(m["max_drawdown"], 4), "profit_factor": round(m["profit_factor"], 3) if m["profit_factor"] != float("inf") else "inf",
        "trades": m["trade_count"], "win_pct": round(m["win_percentage"], 3), "avg_exposure": round(m["average_exposure"], 3),
        "fractional_contribution": [round(c["delta_score"], 3) for c in summary["fractional_contribution"]],
        "risk": summary["risk"],
    }, indent=2))


def cmd_download(args) -> int:
    from .data.calendar import SessionCalendar
    from .data.feed import AlpacaBarFeed
    from .data.store import BarStore

    cfg = build_config(args)
    cal = SessionCalendar.from_config(cfg)
    feed = AlpacaBarFeed(cfg.market.instrument, cal, feed=cfg.alpaca.feed, bar_minutes=int(cfg.market.bar_minutes),
                         adjustment=str(cfg.alpaca.get("adjustment", "split")))
    now = datetime.now(timezone.utc)
    bars = feed.fetch_history(now - timedelta(days=int(args.days)), now)   # forming bar dropped by the feed
    store = BarStore(cfg.market.instrument, int(cfg.market.bar_minutes), bars)
    out = Path(args.out or f"{cfg.paths.data_dir}/{cfg.market.instrument}_{cfg.market.bar_minutes}m.csv")
    store.save(out)
    print(f"saved {len(store)} regular-session bars to {out}")
    return 0


def cmd_paper(args) -> int:
    from .bot import TradingBot
    from .data.feed import AlpacaBarFeed
    from .execution.simulator import AlpacaPaperBroker

    cfg = build_config(args)
    broker = None
    if cfg.alpaca.mirror_orders and not args.no_mirror:
        broker = AlpacaPaperBroker(paper=True)          # the only supported endpoint
    bot = TradingBot(cfg, run_id=args.run_id, artifacts_dir=args.artifacts, broker=broker, log=print, async_retrain=True)
    feed = AlpacaBarFeed(cfg.market.instrument, bot.calendar, feed=cfg.alpaca.feed,
                         bar_minutes=int(cfg.market.bar_minutes), poll_seconds=int(cfg.alpaca.poll_seconds),
                         adjustment=str(cfg.alpaca.get("adjustment", "split")))
    now = datetime.now(timezone.utc)
    history = feed.fetch_history(now - timedelta(days=int(cfg.alpaca.history_days)), now)
    print(f"bootstrapping with {len(history)} completed historical bars (no simulated trading)")
    bot.bootstrap(history)
    if len(bot.store):
        feed.seed_last_timestamp(bot.store.last().timestamp)
    print(f"state={bot.state.value} model={bot.registry.current_version}; entering live loop (Ctrl-C to stop)")
    try:
        run_live_loop(bot, feed, int(cfg.alpaca.poll_seconds), log=print)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        summary = bot.finalize()
        _print_summary(summary)
    return 0


def run_live_loop(bot, feed, poll_seconds: int, log=print, should_stop=None, sleep=time.sleep) -> None:
    """Shared paper-trading loop (CLI and dashboard).

    * every completed bar is processed; in a catch-up batch only the newest bar may queue orders,
    * a feed/API failure never kills the bot: it is audited, the data halt is armed (section 35) and
      polling resumes with back-off,
    * a stale feed inside the session arms the data halt.
    """
    backoff = poll_seconds
    while not (should_stop and should_stop()):
        now = datetime.now(timezone.utc)
        try:
            new_bars = feed.poll_new_bars(now)
            backoff = poll_seconds
        except Exception as exc:  # transient network / API errors
            bot.audit.event("FEED_ERROR", error=f"{type(exc).__name__}: {exc}", now=now.isoformat())
            bot.risk.set_data_halt(True, "market feed unavailable")
            log(f"feed error: {exc}; retrying in {backoff}s")
            sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue
        for i, bar in enumerate(new_bars):
            # Only the newest bar may trade, and only if its successor has not already opened (a bar
            # delivered a full interval late is a catch-up bar: its next open is not a tradable price).
            current = now < bar.close_time + timedelta(minutes=bar.bar_minutes)
            try:
                bot.on_bar(bar, allow_orders=(i == len(new_bars) - 1) and current)
            except Exception as exc:
                bot.audit.event("BAR_ERROR", error=f"{type(exc).__name__}: {exc}", timestamp=bar.timestamp.isoformat())
                bot.risk.set_data_halt(True, f"processing error: {type(exc).__name__}")
                log(f"error processing {bar.timestamp.isoformat()}: {exc}")
                continue
            log(f"{bar.timestamp.isoformat()} close={bar.close:.2f} state={bot.state.value} "
                f"exposure={bot.ledger.exposure:+.3f} equity={bot.ledger.equity:.2f}")
        if bot.validator.is_stale(now) and not bot.risk.data_halted:
            bot.risk.set_data_halt(True, "stale market feed")
            bot.audit.event("DATA_HALT", reason="stale market feed", now=now.isoformat())
        sleep(poll_seconds)


def cmd_train(args) -> int:
    from .bot import TradingBot

    cfg = build_config(args)
    bars = _load_bars(args, cfg)
    bot = TradingBot(cfg, run_id=args.run_id, artifacts_dir=args.artifacts, log=print)
    store = _validated_store(cfg, bars)
    report = bot.trainer.retrain(store, print)
    bot._apply_report(report)
    d = report.to_dict()
    d.pop("grid_results", None)
    print(json.dumps(d, indent=2, default=str))
    bot.finalize()
    return 0 if report.accepted else 1


def cmd_diagnose(args) -> int:
    from .bot import TradingBot
    from .diagnostics.fractional_analysis import FractionalDiagnostics

    cfg = build_config(args)
    bars = _load_bars(args, cfg)
    bot = TradingBot(cfg, run_id=args.run_id, artifacts_dir=args.artifacts, log=print)
    store = _validated_store(cfg, bars)
    window = store.last(bot.trainer.window_bars)
    stationarity = bot.fractional.estimate_stationarity(window.log_close())
    bot.diagnostics.record_stationarity(stationarity)
    print(f"d* = {stationarity.d_star} ({stationarity.selected_by})")
    for c in stationarity.candidates:
        print(f"  d={c.d:.2f} ADF={c.adf_stat:8.3f} p={c.adf_pvalue:.4f} KPSS={c.kpss_stat:7.3f} p={c.kpss_pvalue:.3f} corr={c.correlation:.4f}")
    candidates = [c.d for c in stationarity.candidates][:: max(1, int(args.every))]
    rows = FractionalDiagnostics.oos_score_by_d(bot.trainer, store, candidates, log=print)
    bot.diagnostics.record_oos_by_d(rows)
    bot.diagnostics.save_json()
    plots = bot.diagnostics.plot()
    print("diagnostics written to", bot.diagnostics.dir, "plots:", [str(p) for p in plots])
    bot.audit.close()
    return 0


def cmd_gui(args) -> int:
    from .gui.server import serve

    serve(settings_path=args.settings, host=args.host, port=int(args.port), open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading-bot", description="Fractional-memory systematic trading bot")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", default=None, help="strategy YAML (default: package strategy.yaml)")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (dotted key)")
        sp.add_argument("--symbol", default=None, help="instrument symbol (overrides market.instrument)")
        sp.add_argument("--artifacts", default=None, help="artifacts directory")
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--fast", action="store_true", help="apply strategy_fast.yaml overrides (demo / tests)")

    def data(sp):
        sp.add_argument("--csv", default=None, help="bar history CSV (from `download`)")
        sp.add_argument("--synthetic", type=int, default=None, help="generate N synthetic bars instead of a CSV")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--memory-d", type=float, default=0.40, help="synthetic long-memory order of the stationary component")
        sp.add_argument("--amplitude", type=float, default=3.0, help="synthetic long-memory component amplitude (0 = pure random walk)")

    bt = sub.add_parser("backtest", help="event-driven simulation over stored or synthetic bars")
    common(bt); data(bt)
    bt.add_argument("--max-bars", type=int, default=None)
    bt.add_argument("--quiet", action="store_true")
    bt.add_argument("--require-model", action="store_true", help="exit non-zero unless a model was accepted (CI smoke)")
    bt.set_defaults(func=cmd_backtest)

    dl = sub.add_parser("download", help="download 30-minute bars from Alpaca to CSV")
    common(dl)
    dl.add_argument("--days", type=int, default=1200)
    dl.add_argument("--out", default=None)
    dl.set_defaults(func=cmd_download)

    pp = sub.add_parser("paper", help="live paper trading against Alpaca")
    common(pp)
    pp.add_argument("--no-mirror", action="store_true", help="do not submit mirrored orders to Alpaca")
    pp.set_defaults(func=cmd_paper)

    tr = sub.add_parser("train", help="run one retraining cycle and print the report")
    common(tr); data(tr)
    tr.set_defaults(func=cmd_train)

    dg = sub.add_parser("diagnose", help="stationarity and out-of-sample score curves versus d")
    common(dg); data(dg)
    dg.add_argument("--every", type=int, default=2, help="evaluate every k-th candidate d for the OOS curve")
    dg.set_defaults(func=cmd_diagnose)

    gui = sub.add_parser("gui", help="local browser dashboard (settings, API keys, run control, live status)")
    gui.add_argument("--settings", default="settings.json", help="where the dashboard stores settings and API keys")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", default=8765, type=int)
    gui.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    gui.set_defaults(func=cmd_gui)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
