import warnings

import pytest

from trading_bot.config import load_config
from trading_bot.data.calendar import SessionCalendar
from trading_bot.data.store import BarStore
from trading_bot.data.synthetic import generate_synthetic_bars
from trading_bot.features.engine import FeatureEngine
from trading_bot.fractional.engine import FractionalEngine

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={"market": {"instrument": "SYN"}})


@pytest.fixture(scope="session")
def calendar(cfg):
    return SessionCalendar.from_config(cfg)


@pytest.fixture(scope="session")
def fractional(cfg):
    return FractionalEngine.from_config(cfg)


@pytest.fixture(scope="session")
def bars_1500(calendar):
    return generate_synthetic_bars(1500, seed=11, instrument="SYN", calendar=calendar)


@pytest.fixture(scope="session")
def store_1500(bars_1500):
    return BarStore("SYN", 30, bars_1500)


@pytest.fixture()
def feature_engine(cfg, fractional, calendar):
    return FeatureEngine(cfg, fractional, calendar, adaptive_d=0.35)


from trading_bot.tests.test_training_and_bot import bot_run, fast_cfg  # noqa: E402,F401  (shared fixtures)
