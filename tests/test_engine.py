import inspect
import numpy as np
import pandas as pd
import pytest
import strategies
from engine import Backtester

def make_data(closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({'open': c, 'high': c * 1.01, 'low': c * 0.99, 'close': c, 'volume': np.full(len(c), 1000)})

class AlwaysLong:
    def generate_signals(self, data):
        return [1] * len(data)

class BuyThenSell:
    def generate_signals(self, data):
        s = [0] * len(data)
        s[1], s[3] = 1, -1
        return s

@pytest.mark.parametrize('interval,market,expected', [
    ('1d', 'equity', 252), ('1d', 'crypto', 365), ('1h', 'equity', 1638), ('1h', 'crypto', 8760)])
def test_bars_per_year(interval, market, expected):
    assert Backtester.bars_per_year(interval, market) == expected

def test_signal_executes_on_next_bar():
    bt = Backtester(make_data([100] * 6), BuyThenSell(), commission=0, slippage=0)
    bt.run_fast()
    assert [t['i'] for t in bt.trades] == [2, 4]

def test_flat_price_round_trip_costs_exactly_fees_and_slippage():
    bt = Backtester(make_data([100] * 6), BuyThenSell(), initial_capital=10000,
                    commission=0.001, slippage=0.0005, position_size_pct=0.95)
    bt.run_fast()
    shares = 10000 * 0.95 / 100
    expected = 10000 - shares * (100 * 1.0005 * 1.001 - 100 * 0.9995 * 0.999)
    assert bt.capital == pytest.approx(expected)

def test_always_long_matches_price_return_without_costs():
    closes = np.linspace(100, 200, 100)
    bt = Backtester(make_data(closes), AlwaysLong(), commission=0, slippage=0, position_size_pct=0.5)
    bt.run_fast()
    assert bt.capital == pytest.approx(5000 + 5000 * closes[-1] / closes[1])

STRATEGY_CLASSES = [c for _, c in inspect.getmembers(strategies, inspect.isclass) if hasattr(c, 'generate_signals')]

@pytest.mark.parametrize('cls', STRATEGY_CLASSES)
def test_every_strategy_returns_one_valid_signal_per_bar(cls):
    rng = np.random.default_rng(0)
    data = make_data(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    signals = cls().generate_signals(data.copy())
    assert len(signals) == len(data)
    assert set(np.unique(signals)) <= {-1, 0, 1}
