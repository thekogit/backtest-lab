import yfinance as yf
import pandas as pd
import numpy as np
import inspect
import json
from datetime import datetime
import strategies
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import pickle
from pathlib import Path
import multiprocessing
from engine import Backtester
from report_generator import ReportGenerator

print_lock = Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

class DataCache:
    def __init__(self, cache_dir='./cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_lock = Lock()
        self.download_lock = Lock()
        self.last_download_time = {}
        self.min_request_interval = 0.25

    def get_cache_path(self, symbol, period, interval):
        """Generate cache file path"""
        filename = f"{symbol}_{period}_{interval}.pkl"
        return self.cache_dir / filename

    def load_from_disk(self, symbol, period, interval):
        """Load from disk cache"""
        cache_path = self.get_cache_path(symbol, period, interval)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    safe_print(f"Cache: {symbol} {interval}")
                    return data
            except Exception:
                pass
        return None

    def save_to_disk(self, symbol, period, interval, data):
        cache_path = self.get_cache_path(symbol, period, interval)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def fetch_with_rate_limit(self, symbol, period, interval):
        disk_data = self.load_from_disk(symbol, period, interval)
        if disk_data is not None:
            with self.cache_lock:
                self.memory_cache[f"{symbol}_{period}_{interval}"] = disk_data.copy()
            return disk_data.copy()

        cache_key = f"{symbol}_{period}_{interval}"
        with self.cache_lock:
            if cache_key in self.memory_cache:
                safe_print(f"Memory: {symbol} {interval}")
                return self.memory_cache[cache_key].copy()

        with self.download_lock:
            with self.cache_lock:
                if cache_key in self.memory_cache:
                    return self.memory_cache[cache_key].copy()

            last_time = self.last_download_time.get('last_download', 0)
            elapsed = time.time() - last_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)

            safe_print(f"Download: {symbol} {interval}")

            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    period=period,
                    interval=interval,
                    prepost=False,
                    auto_adjust=True,
                    timeout=10
                )

                if df.empty or len(df) < 30:
                    safe_print(f"No data: {symbol} {interval}")
                    return None

                df.reset_index(inplace=True)
                df.columns = df.columns.str.lower()

                if 'date' in df.columns:
                    df.rename(columns={'date': 'timestamp'}, inplace=True)
                elif 'datetime' in df.columns:
                    df.rename(columns={'datetime': 'timestamp'}, inplace=True)

                result = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

                with self.cache_lock:
                    self.memory_cache[cache_key] = result.copy()
                self.save_to_disk(symbol, period, interval, result)

                self.last_download_time['last_download'] = time.time()
                safe_print(f"Downloaded: {symbol} {interval} ({len(result)} bars)")

                return result

            except Exception as e:
                safe_print(f"Error: {symbol} {interval}: {str(e)[:40]}")
                return None

    def prefetch(self, assets, intervals_config):
        safe_print("\n🔄 Pre-fetching all data (rate-limited)...")
        safe_print("   First run downloads, future runs use cache\n")

        total = len(assets) * len(intervals_config)
        count = 0
        downloaded = 0
        cached = 0

        for asset in assets:
            symbol = asset['symbol']

            for interval_config in intervals_config:
                count += 1
                interval = interval_config['interval']
                period  = interval_config['period']

                prev_ts = self.last_download_time.get('last_download', 0)

                data = self.fetch_with_rate_limit(symbol, period, interval)
                if data is None:
                    pass
                else:
                    new_ts = self.last_download_time.get('last_download', 0)
                    if new_ts != prev_ts:
                        downloaded += 1
                    else:
                        cached += 1

                if count % 10 == 0:
                    safe_print(f"   Progress: {count}/{total} ({cached} cached, {downloaded} downloaded)")

        safe_print(f"\n✅ Pre-fetch complete!")
        safe_print(f"   Total: {total} | Cached: {cached} | Downloaded: {downloaded}\n")


def get_all_strategies():
    strategy_classes = []
    for name in dir(strategies):
        obj = getattr(strategies, name)
        if inspect.isclass(obj) and hasattr(obj, 'generate_signals'):
            strategy_classes.append(obj)
    return strategy_classes


def run_combination(args):
    asset, interval_config, strategy_cls, cache = args

    symbol = asset['symbol']
    interval = interval_config['interval']
    period = interval_config['period']
    strategy = strategy_cls()
    strategy_name = strategy_cls.__name__

    try:
        data = cache.fetch_with_rate_limit(symbol, period, interval)

        if data is None or len(data) < 30:
            return None

        backtester = Backtester(data, strategy, interval=interval, market=asset['type'])
        portfolio_values = backtester.run_fast()
        if portfolio_values is None:
            return None

        split = int(len(data) * 0.7)
        is_metrics = backtester.calculate_performance_fast(portfolio_values, 0, split)
        metrics = backtester.calculate_performance_fast(portfolio_values, split, None)
        if not metrics or not is_metrics:
            return None

        return_pct = metrics['total_return'] * 100

        if return_pct > 20:
            status = "🟢"
        elif return_pct > 0:
            status = "🟡"
        else:
            status = "🔴"

        safe_print(f"  {status} {symbol:10s} | {interval:4s} | {strategy_name:20s} | R: {return_pct:7.2f}% | Excess: {metrics['excess_vs_buy_hold']*100:7.2f}%")

        return {
            'symbol': symbol,
            'asset_name': asset['name'],
            'asset_type': asset['type'],
            'interval': interval,
            'strategy_name': strategy_name,
            'metrics': metrics,
            'is_metrics': is_metrics
        }

    except Exception as e:
        safe_print(f"FAILED {strategy_cls.__name__} {symbol} {interval}: {e!r}")
        return None


def print_oos_table(results):
    best = {}
    for r in results:
        key = (r['symbol'], r['interval'])
        if key not in best or r['is_metrics']['sharpe_ratio'] > best[key]['is_metrics']['sharpe_ratio']:
            best[key] = r
    print("| Asset | Interval | Picked on in-sample | IS Sharpe | OOS Sharpe | OOS return | Buy & hold OOS |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(best.values(), key=lambda r: (r['interval'], r['symbol'])):
        m, im = r['metrics'], r['is_metrics']
        bh = m['total_return'] - m['excess_vs_buy_hold']
        print(f"| {r['symbol']} | {r['interval']} | {r['strategy_name']} | {im['sharpe_ratio']:.2f} | "
              f"{m['sharpe_ratio']:.2f} | {m['total_return']*100:.1f}% | {bh*100:.1f}% |")


def run_cached_backtest(assets_file='assets.json',
                              test_all_intervals=True,
                              max_workers=None):

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    with open(assets_file, 'r') as f:
        config = json.load(f)

    assets = config['assets']
    intervals_config = config['settings']['intervals']

    if not test_all_intervals:
        intervals_config = [intervals_config[-1]]

    all_strategies = get_all_strategies()
    cache = DataCache()

    safe_print("\n" + "="*70)
    safe_print("CACHED BACKTESTER v3")
    safe_print("="*70)
    safe_print(f"Assets: {len(assets)}")
    safe_print(f"Intervals: {len(intervals_config)}")
    safe_print(f"Strategies: {len(all_strategies)}")
    safe_print(f"Threads: {max_workers}")
    safe_print(f"Cache: ./cache/")
    safe_print("="*70)

    cache.prefetch(assets, intervals_config)

    tasks = []
    for asset in assets:
        for interval_config in intervals_config:
            for strategy_cls in all_strategies:
                tasks.append((asset, interval_config, strategy_cls, cache))

    results = []
    completed = 0
    total = len(tasks)

    safe_print(f"⚡ Testing {total} combinations on cached data...\n")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_combination, task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            result = future.result()

            if result:
                results.append(result)

            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate > 0 else 0
                safe_print(f"\n📊 [{completed}/{total}] Rate: {rate:.1f}/s | ETA: {remaining:.0f}s\n")

    elapsed_time = time.time() - start_time

    safe_print("\n" + "="*70)
    safe_print("BACKTEST COMPLETE!")
    safe_print("="*70)
    safe_print(f"Time: {elapsed_time:.1f}s")
    safe_print(f"Tests: {len(results)}/{total}")
    safe_print(f"Rate: {total/elapsed_time:.1f} tests/sec")
    safe_print(f"Cache: ./cache/")
    safe_print("="*70)

    return results

if __name__ == "__main__":

    results = run_cached_backtest(
        assets_file='assets.json',
        test_all_intervals=True,
        max_workers=None
    )
    if results:
        report_gen = ReportGenerator(output_dir='./reports')
        report_gen.generate(results, output_file='backtest_report')
        print_oos_table(results)
        print("\n DONE!")