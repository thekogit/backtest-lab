import pandas as pd
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MACDStrategy:
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data):
        macd = ta.trend.MACD(data['close'],
                            window_fast=self.fast,
                            window_slow=self.slow,
                            window_sign=self.signal)
        data['macd'] = macd.macd()
        data['signal'] = macd.macd_signal()

        signals = [0] * len(data)
        for i in range(1, len(data)):
            if data['macd'].iloc[i] > data['signal'].iloc[i] and \
               data['macd'].iloc[i-1] <= data['signal'].iloc[i-1]:
                signals[i] = 1  # Buy
            elif data['macd'].iloc[i] < data['signal'].iloc[i] and \
                 data['macd'].iloc[i-1] >= data['signal'].iloc[i-1]:
                signals[i] = -1  # Sell

        return signals


class RSIStrategy:
    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data):
        data['rsi'] = ta.momentum.RSIIndicator(data['close'],
                                              window=self.period).rsi()

        signals = [0] * len(data)
        for i in range(1, len(data)):
            if data['rsi'].iloc[i] < self.oversold:
                signals[i] = 1  # Buy
            elif data['rsi'].iloc[i] > self.overbought:
                signals[i] = -1  # Sell

        return signals


class CombinedMACDRSI:
    def __init__(self):
        self.macd = MACDStrategy(fast=3, slow=10, signal=16)
        self.rsi = RSIStrategy(period=14, oversold=20, overbought=80)

    def generate_signals(self, data):
        macd_signals = self.macd.generate_signals(data.copy())
        rsi_signals = self.rsi.generate_signals(data.copy())

        combined = [0] * len(data)
        for i in range(len(data)):
            if macd_signals[i] == 1 and rsi_signals[i] == 1:
                combined[i] = 1
            elif macd_signals[i] == -1 and rsi_signals[i] == -1:
                combined[i] = -1

        return combined


class TabularQLearning:
    def __init__(self, lookback=20, learning_rate=0.01):
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.q_table = {}

    def _discretize_state(self, returns, volatility):
        return (round(returns, 2), round(volatility, 3))

    def generate_signals(self, data):
        if len(data) < self.lookback:
            return [0] * len(data)

        signals = [0] * len(data)
        closes = data['close'].values

        returns = np.diff(closes) / closes[:-1] * 100
        returns = np.insert(returns, 0, 0)

        volatility = np.zeros(len(closes))
        for i in range(self.lookback, len(closes)):
            volatility[i] = np.std(returns[i-self.lookback:i])

        for i in range(self.lookback, len(closes)):
            state = self._discretize_state(returns[i], volatility[i])

            if state not in self.q_table:
                self.q_table[state] = [0, 0, -1]

            q_values = self.q_table[state]
            best_action = np.argmax(q_values)

            if best_action == 1:
                signals[i] = 1
            elif best_action == 2:
                signals[i] = -1

            if i + 1 < len(closes):
                future_return = (closes[i+1] - closes[i]) / closes[i]
                reward = future_return * 100
                old_value = q_values[best_action]
                new_value = old_value + self.learning_rate * reward
                q_values[best_action] = new_value

        return signals


class MomentumVolFilter:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _calculate_features(self, data):
        closes = data['close'].values
        features = []

        for i in range(len(closes)):
            if i < self.lookback:
                features.append([0, 0, 0])
                continue

            momentum = (closes[i] - closes[i-self.lookback]) / closes[i-self.lookback]
            returns = np.diff(closes[i-self.lookback:i]) / closes[i-self.lookback:i-1]
            volatility = np.std(returns)
            trend = np.mean(returns)

            features.append([momentum, volatility, trend])

        return np.array(features)

    def generate_signals(self, data):
        features = self._calculate_features(data)
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            momentum, volatility, trend = features[i]

            if momentum > 0 and trend > 0 and volatility < 0.05:
                signals[i] = 1
            elif momentum < -0.02 and trend < 0:
                signals[i] = -1
            elif momentum > 0.03 and volatility > 0.1:
                signals[i] = -1

        return signals


class MovingAverageCrossover:
    def __init__(self, fast=20, slow=50, use_ema=False):
        self.fast = fast
        self.slow = slow
        self.use_ema = use_ema

    def generate_signals(self, data):
        if self.use_ema:
            data['fast_ma'] = data['close'].ewm(span=self.fast, adjust=False).mean()
            data['slow_ma'] = data['close'].ewm(span=self.slow, adjust=False).mean()
        else:
            data['fast_ma'] = data['close'].rolling(window=self.fast).mean()
            data['slow_ma'] = data['close'].rolling(window=self.slow).mean()

        signals = [0] * len(data)
        for i in range(1, len(data)):
            if data['fast_ma'].iloc[i] > data['slow_ma'].iloc[i] and \
               data['fast_ma'].iloc[i-1] <= data['slow_ma'].iloc[i-1]:
                signals[i] = 1
            elif data['fast_ma'].iloc[i] < data['slow_ma'].iloc[i] and \
                 data['fast_ma'].iloc[i-1] >= data['slow_ma'].iloc[i-1]:
                signals[i] = -1

        return signals


class MomentumStrategy:
    def __init__(self, lookback=14, threshold=0.02):
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data):
        closes = data['close'].values
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            roc = (closes[i] - closes[i-self.lookback]) / closes[i-self.lookback]

            if roc > self.threshold:
                signals[i] = 1
            elif roc < -self.threshold:
                signals[i] = -1

        return signals


class ZScoreMeanReversion:
    def __init__(self, lookback=30, entry_threshold=2.0, exit_threshold=0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold 
        self.exit_threshold = exit_threshold

    def generate_signals(self, data):
        data['ma'] = data['close'].rolling(window=self.lookback).mean()
        data['std'] = data['close'].rolling(window=self.lookback).std()

        data['zscore'] = (data['close'] - data['ma']) / data['std']

        signals = [0] * len(data)
        position = 0

        for i in range(self.lookback, len(data)):
            if pd.notna(data['zscore'].iloc[i]):
                zscore = data['zscore'].iloc[i]

                if zscore > self.entry_threshold and position != -1:
                    signals[i] = -1
                    position = -1
                elif zscore < -self.entry_threshold and position != 1:
                    signals[i] = 1
                    position = 1
                elif abs(zscore) < self.exit_threshold:
                    if position == 1:
                        signals[i] = -1
                        position = 0
                    elif position == -1:
                        signals[i] = 1
                        position = 0

        return signals


class LowVolTrend:
    def __init__(self, lookback=20, volatility_threshold=0.02):
        self.lookback = lookback
        self.volatility_threshold = volatility_threshold

    def generate_signals(self, data):
        closes = data['close'].values
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            returns = np.diff(closes[i-self.lookback:i]) / closes[i-self.lookback:i-1]
            avg_return = np.mean(returns)
            volatility = np.std(returns)

            if avg_return > 0 and volatility < self.volatility_threshold:
                signals[i] = 1
            elif avg_return < 0 or volatility > self.volatility_threshold * 2:
                signals[i] = -1

        return signals


class VolRegimeSwitch:
    def __init__(self, lookback=10, volatility_threshold=0.03):
        self.lookback = lookback
        self.volatility_threshold = volatility_threshold

    def generate_signals(self, data):
        closes = data['close'].values
        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            returns = np.diff(closes[i-self.lookback:i]) / closes[i-self.lookback:i-1]
            realized_vol = np.std(returns)

            if i >= self.lookback * 2:
                hist_returns = np.diff(closes[i-self.lookback*2:i-self.lookback]) / \
                              closes[i-self.lookback*2:i-self.lookback-1]
                historical_vol = np.std(hist_returns)

                if realized_vol < historical_vol * 0.7:
                    signals[i] = 1
                elif realized_vol > historical_vol * 1.3:
                    signals[i] = -1

        return signals


class VolumeSpikeMomentum:
    def __init__(self, lookback=20, volume_threshold=1.5):
        self.lookback = lookback
        self.volume_threshold = volume_threshold

    def generate_signals(self, data):
        data['volume_ma'] = data['volume'].rolling(window=self.lookback).mean()

        data['price_change'] = data['close'].pct_change(self.lookback)

        signals = [0] * len(data)

        for i in range(self.lookback, len(data)):
            if pd.notna(data['volume_ma'].iloc[i]) and pd.notna(data['price_change'].iloc[i]):
                volume_ratio = data['volume'].iloc[i] / data['volume_ma'].iloc[i]
                price_change = data['price_change'].iloc[i]

                if volume_ratio > self.volume_threshold and price_change > 0.02:
                    signals[i] = 1
                elif volume_ratio > self.volume_threshold and price_change < -0.02:
                    signals[i] = -1

        return signals


class BollingerBandsStrategy:
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data):
        data['bb_middle'] = data['close'].rolling(window=self.window).mean()
        data['bb_std'] = data['close'].rolling(window=self.window).std()
        data['bb_upper'] = data['bb_middle'] + (self.num_std * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (self.num_std * data['bb_std'])
        
        signals = [0] * len(data)
        
        for i in range(1, len(data)):
            if pd.notna(data['bb_upper'].iloc[i]) and pd.notna(data['bb_lower'].iloc[i]):
                if data['close'].iloc[i] <= data['bb_lower'].iloc[i]:
                    signals[i] = 1
                elif data['close'].iloc[i] >= data['bb_upper'].iloc[i]:
                    signals[i] = -1
        
        return signals


class MomentumLowVolEnsemble:
    def __init__(self, 
                 momentum_lookback=14,
                 momentum_threshold=0.02,
                 low_vol_lookback=20,
                 volatility_threshold=0.02):
        self.momentum_lookback = momentum_lookback
        self.momentum_threshold = momentum_threshold
        self.low_vol_lookback = low_vol_lookback
        self.volatility_threshold = volatility_threshold
        self.momentum_weight = 0.5
        self.low_vol_weight = 0.5
    
    def generate_signals(self, data):
        momentum_signals = self._momentum_signals(data)
        low_vol_signals = self._low_vol_signals(data)
        
        signals = [0] * len(data)
        for i in range(len(data)):
            if momentum_signals[i] == 1 and low_vol_signals[i] == 1:
                signals[i] = 1
            elif momentum_signals[i] == -1 or low_vol_signals[i] == -1:
                signals[i] = -1
        
        return signals
    
    def _momentum_signals(self, data):
        signals = [0] * len(data)
        closes = data['close'].values
        
        for i in range(self.momentum_lookback, len(data)):
            roc = (closes[i] - closes[i-self.momentum_lookback]) / closes[i-self.momentum_lookback]
            
            if roc > self.momentum_threshold:
                signals[i] = 1
            elif roc < -self.momentum_threshold:
                signals[i] = -1
        
        return signals
    
    def _low_vol_signals(self, data):
        signals = [0] * len(data)
        closes = data['close'].values
        
        for i in range(self.low_vol_lookback, len(data)):
            returns = np.diff(closes[i-self.low_vol_lookback:i]) / closes[i-self.low_vol_lookback:i-1]
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            
            if avg_return > 0 and volatility < self.volatility_threshold:
                signals[i] = 1
            elif avg_return < 0 or volatility > self.volatility_threshold * 2:
                signals[i] = -1
        
        return signals
