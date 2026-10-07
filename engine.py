import numpy as np
import pandas as pd


class Backtester:

    __slots__ = ['data', 'strategy', 'interval', 'market', 'initial_capital', 'commission', 'slippage',
                 'position_size_pct', 'capital', 'position', 'entry_price', 
                 'trades', 'signals', 'risk_free_annual']

    def __init__(self, data, strategy, interval='1d', market='equity', initial_capital=10000, 
                 commission=0.001, slippage=0.0005, position_size_pct=0.95, risk_free_annual=0.04):

        self.data = data
        self.strategy = strategy
        self.interval = interval
        self.market = market
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.signals = []
        self.risk_free_annual = risk_free_annual

    def run_fast(self):
        try:
            signals = np.nan_to_num(np.asarray(self.strategy.generate_signals(self.data)), nan=0.0)
            signals = np.sign(signals)
        except Exception:
            return None
        
        signals = np.roll(signals, 1)
        signals[0] = 0

        warmup = getattr(self.strategy, 'warmup_bars', 0)
        if warmup > 0:
            signals[:warmup] = 0
        
        self.signals = signals

        closes = self.data['close'].values
        n = len(closes)
        portfolio_values = np.zeros(n)

        for i in range(n):
            close_price = closes[i]
            signal = self.signals[i]

            portfolio_values[i] = self.capital + (self.position * close_price)

            if signal == 1 and self.position == 0:
                shares_to_buy = (self.capital * self.position_size_pct) / close_price
                if shares_to_buy > 0:
                    execution_price = close_price * (1 + self.slippage)
                    cost = shares_to_buy * execution_price * (1 + self.commission)
                    if cost <= self.capital:
                        self.trades.append({'i': i, 'side': 'buy', 'price': execution_price, 'size': shares_to_buy})
                        self.position = shares_to_buy
                        self.capital -= cost
                        self.entry_price = execution_price

            elif signal == -1 and self.position > 0:
                execution_price = close_price * (1 - self.slippage)
                proceeds = self.position * execution_price * (1 - self.commission)
                self.trades.append({'i': i, 'side': 'sell', 'price': execution_price, 'size': self.position})
                self.capital += proceeds
                self.position = 0

        if self.position > 0:
            execution_price = closes[-1] * (1 - self.slippage)
            proceeds = self.position * execution_price * (1 - self.commission)
            self.trades.append({'i': n-1, 'side': 'sell', 'price': execution_price, 'size': self.position})
            self.capital += proceeds
            self.position = 0

        return portfolio_values

    @staticmethod
    def bars_per_year(interval, market='equity'):
        if market == 'crypto':
            days_per_year = 365
            hours_per_day = 24
            mins_per_day = 24 * 60
        else:
            days_per_year = 252
            hours_per_day = 6.5
            mins_per_day = 390

        if interval.endswith('d'):
            return days_per_year
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return int((hours_per_day / hours) * days_per_year)
        elif interval.endswith('m'):
            mins = int(interval[:-1])
            return int((mins_per_day / mins) * days_per_year)

        return days_per_year

    def calculate_performance_fast(self, portfolio_values, start_index=0, end_index=None):
        if portfolio_values is None:
            return None
        pv = np.asarray(portfolio_values[start_index:end_index], dtype=float)
        if len(pv) < 2 or pv[0] <= 0:
            return None
        total_return = pv[-1] / pv[0] - 1
        cummax = np.maximum.accumulate(pv)
        max_drawdown = float(np.min((pv - cummax) / cummax))
        returns = np.diff(pv) / pv[:-1]
        returns = returns[np.isfinite(returns)]
        bpyr = Backtester.bars_per_year(self.interval, self.market)
        rf_bar = (1 + self.risk_free_annual) ** (1 / bpyr) - 1
        excess = returns - rf_bar
        sd = np.std(excess, ddof=1) if len(excess) > 1 else 0.0
        sharpe = float(np.mean(excess) / sd * np.sqrt(bpyr)) if sd > 0 else 0.0
        closes = self.data['close'].values[start_index:end_index]
        buy_hold_return = closes[-1] / closes[0] - 1
        stop = len(portfolio_values) if end_index is None else end_index
        num_trades = sum(1 for t in self.trades if t['side'] == 'sell' and start_index <= t['i'] < stop)
        return {
            'total_return': float(total_return),
            'final_value': float(pv[-1]),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'excess_vs_buy_hold': float(total_return - buy_hold_return),
            'num_trades': num_trades,
        }
