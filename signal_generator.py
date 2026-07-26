"""
Multi-Strategy Daily Signal Generator
=====================================
Ranks multiple trading strategies by rolling walk-forward performance
(Sharpe, profit factor, expectancy) and surfaces the current best one's
live signal per asset.

- Weekdays: Gold (XAUUSD) + Silver (XAGUSD)
- Weekends: Crypto watchlist (BTC, ETH, SOL, AAVE, AVAX, ICP, TON, HYPE)

Requirements:
    pip install pandas numpy requests MetaTrader5

Usage:
    python signal_generator.py --demo
    python signal_generator.py
    python signal_generator.py --crypto-source mt5
    python signal_generator.py --crypto-source fmp
    python signal_generator.py --lookback 90 --folds 4
    python signal_generator.py --telegram
    python signal_generator.py --email

Environment variables:
    Telegram
        SIGNAL_TELEGRAM_BOT_TOKEN
        SIGNAL_TELEGRAM_CHAT_ID

    Email
        SIGNAL_SMTP_SERVER
        SIGNAL_SMTP_PORT
        SIGNAL_SMTP_USER
        SIGNAL_SMTP_PASSWORD
        SIGNAL_FROM_EMAIL
        SIGNAL_TO_EMAIL

    FMP (optional)
        FMP_API_KEY

Notes:
    - Symbol names must exactly match your broker's MT5 Market Watch listing.
    - This generates signals only; it never places trades.
    - Backtested performance does not guarantee future results.
      This is not financial advice — use your own risk management.
"""

import os
import json
import datetime
import argparse
import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
METALS = ['XAUUSD', 'XAGUSD']
CRYPTO = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AAVEUSD', 'AVAXUSD', 'ICPUSD', 'TONUSD', 'HYPEUSD']

BINANCE_SYMBOL_MAP = {
    'BTCUSD': 'BTCUSDT', 'ETHUSD': 'ETHUSDT', 'SOLUSD': 'SOLUSDT',
    'AAVEUSD': 'AAVEUSDT', 'AVAXUSD': 'AVAXUSDT', 'ICPUSD': 'ICPUSDT',
    'TONUSD': 'TONUSDT', 'HYPEUSD': 'HYPEUSDT',
}

FMP_SYMBOL_MAP = {
    'XAUUSD': 'XAUUSD',
    'XAGUSD': 'XAGUSD',
    'BTCUSD': 'BTCUSD',
    'ETHUSD': 'ETHUSD',
    'SOLUSD': 'SOLUSD',
    'AAVEUSD': 'AAVEUSD',
    'AVAXUSD': 'AVAXUSD',
    'ICPUSD': 'ICPUSD',
    'TONUSD': 'TONUSD',
    'HYPEUSD': 'HYPEUSD',
}

DEFAULT_SIGNAL_JOURNAL_PATH = 'signal_journal.csv'
DEFAULT_LEARNING_STATE_PATH = 'learning_state.json'
DEFAULT_METALS_START_HOUR = 4
DEFAULT_METALS_END_HOUR = 18
DEFAULT_OTHER_MIN_CONF = 0.90
DEFAULT_OTHER_MAX_CONF = 0.98
DEFAULT_MIN_ATR_PCT_MOVING = 0.004


def get_today_universe(reference_date=None):
    d = reference_date or datetime.date.today()
    return CRYPTO if d.weekday() >= 5 else METALS


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(series, period=20, num_std=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def donchian(df, period=20):
    return df['high'].rolling(period).max(), df['low'].rolling(period).min()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def strategy_trend_following(df, fast=20, slow=50):
    f, s = ema(df['close'], fast), ema(df['close'], slow)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[f > s] = 1
    sig[f < s] = -1
    return sig.shift(1).fillna(0)


def strategy_mean_reversion(df, rsi_period=14, bb_period=20, bb_std=2):
    r = rsi(df['close'], rsi_period)
    upper, _, lower = bollinger_bands(df['close'], bb_period, bb_std)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[(r < 30) & (df['close'] <= lower)] = 1
    sig[(r > 70) & (df['close'] >= upper)] = -1
    return sig.shift(1).fillna(0)


def strategy_momentum(df, roc_period=10, threshold=0.01):
    roc = df['close'].pct_change(roc_period)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[roc > threshold] = 1
    sig[roc < -threshold] = -1
    return sig.shift(1).fillna(0)


def strategy_volatility_breakout(df, don_period=20, atr_period=14):
    upper, lower = donchian(df, don_period)
    a = atr(df, atr_period)
    a_ma = a.rolling(atr_period).mean()
    sig = pd.Series(0.0, index=df.index)
    sig[(df['close'] > upper.shift(1)) & (a > a_ma)] = 1
    sig[(df['close'] < lower.shift(1)) & (a > a_ma)] = -1
    sig = sig.replace(0, np.nan).ffill().fillna(0)
    return sig.shift(1).fillna(0)


STRATEGIES = {
    'trend_following': strategy_trend_following,
    'mean_reversion': strategy_mean_reversion,
    'momentum': strategy_momentum,
    'volatility_breakout': strategy_volatility_breakout,
}


# ---------------------------------------------------------------------------
# Backtest + metrics
# ---------------------------------------------------------------------------
def backtest(df, signal, transaction_cost_bps=3.0, slippage_bps=1.0):
    out = df.copy()
    out['position'] = signal.values
    out['returns'] = out['close'].pct_change().fillna(0)

    turnover = out['position'].diff().abs().fillna(out['position'].abs())
    cost_per_turn = (transaction_cost_bps + slippage_bps) / 10000.0
    costs = turnover * cost_per_turn

    out['strategy_returns'] = out['position'] * out['returns'] - costs
    out['equity'] = (1 + out['strategy_returns']).cumprod()
    out['turnover'] = turnover
    return out


def compute_metrics(df, periods_per_year=252):
    rets = df['strategy_returns'].dropna()
    active = rets[rets != 0]

    if len(rets) < 10 or rets.std() == 0:
        return {
            'sharpe': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'trades': 0,
            'turnover': 0.0,
            'score': -999.0,
        }

    sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year)
    wins, losses = rets[rets > 0], rets[rets < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 10.0
    expectancy = rets.mean()
    win_rate = len(wins) / len(active) if len(active) else 0.0

    equity = df['equity'].dropna()
    dd = (equity - equity.cummax()) / equity.cummax()
    max_dd = float(dd.min()) if len(dd) else 0.0

    turnover = float(df['turnover'].mean()) if 'turnover' in df else 0.0

    metrics = {
        'sharpe': round(float(sharpe), 3),
        'profit_factor': round(float(min(profit_factor, 10.0)), 3),
        'expectancy': round(float(expectancy), 6),
        'win_rate': round(float(win_rate), 3),
        'max_drawdown': round(max_dd, 3),
        'trades': int((df['position'].diff().fillna(0) != 0).sum()),
        'turnover': round(turnover, 4),
    }

    drawdown_score = max(0.0, 1.0 + metrics['max_drawdown'])
    inactivity_penalty = 0.25 if metrics['trades'] < 4 else 0.0
    overtrading_penalty = min(metrics['turnover'] * 2.0, 0.5)

    metrics['score'] = round(
        (0.35 * metrics['sharpe'])
        + (0.25 * metrics['profit_factor'])
        + (0.20 * (metrics['expectancy'] * 1000.0))
        + (0.10 * (metrics['win_rate'] * 10.0))
        + (0.10 * (drawdown_score * 5.0))
        - inactivity_penalty
        - overtrading_penalty,
        3,
    )
    return metrics


def _aggregate_fold_metrics(fold_metrics):
    if not fold_metrics:
        return {
            'sharpe': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'trades': 0,
            'turnover': 0.0,
            'score': -999.0,
            'stability': 0.0,
            'folds_used': 0,
        }

    score_list = [m['score'] for m in fold_metrics]
    stability = 1.0 / (1.0 + np.std(score_list))

    agg = {
        'sharpe': round(float(np.median([m['sharpe'] for m in fold_metrics])), 3),
        'profit_factor': round(float(np.median([m['profit_factor'] for m in fold_metrics])), 3),
        'expectancy': round(float(np.median([m['expectancy'] for m in fold_metrics])), 6),
        'win_rate': round(float(np.median([m['win_rate'] for m in fold_metrics])), 3),
        'max_drawdown': round(float(np.median([m['max_drawdown'] for m in fold_metrics])), 3),
        'trades': int(np.median([m['trades'] for m in fold_metrics])),
        'turnover': round(float(np.median([m['turnover'] for m in fold_metrics])), 4),
        'score': round(float(np.median(score_list) + 0.2 * stability), 3),
        'stability': round(float(stability), 3),
        'folds_used': len(fold_metrics),
    }
    return agg


def rank_strategies(
    df,
    lookback=90,
    periods_per_year=252,
    folds=3,
    transaction_cost_bps=3.0,
    slippage_bps=1.0,
):
    warmup = 120
    window = df.tail(lookback + warmup).reset_index(drop=True)
    eval_window = window.tail(lookback)

    results = {}
    for name, fn in STRATEGIES.items():
        sig_all = fn(window)
        bt_all = backtest(
            window,
            sig_all,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        bt_eval = bt_all.tail(lookback).reset_index(drop=True)

        folds = max(1, min(folds, len(bt_eval) // 15))
        fold_metrics = []
        for idx_chunk in np.array_split(bt_eval.index.to_numpy(), folds):
            fold_df = bt_eval.loc[idx_chunk]
            if len(fold_df) < 10:
                continue
            fold_metrics.append(compute_metrics(fold_df, periods_per_year=periods_per_year))

        results[name] = _aggregate_fold_metrics(fold_metrics)

    return sorted(results.items(), key=lambda kv: kv[1]['score'], reverse=True)


# ---------------------------------------------------------------------------
# Live signal
# ---------------------------------------------------------------------------
def _signal_confidence(best_score, second_score, best_sharpe, best_stability):
    spread = best_score - second_score
    base = 0.5 + np.tanh(spread / 2.0) * 0.3
    sharpe_boost = np.tanh(best_sharpe / 2.0) * 0.1
    stability_boost = max(0.0, min(best_stability, 1.0)) * 0.1
    return round(float(max(0.0, min(1.0, base + sharpe_boost + stability_boost))), 3)


def get_live_signal(df, strategy_name, atr_stop_mult=1.5, atr_target_mult=2.5):
    sig = STRATEGIES[strategy_name](df)
    direction_code = sig.iloc[-1]
    price = float(df['close'].iloc[-1])
    a = float(atr(df).iloc[-1])
    direction = 'BUY' if direction_code == 1 else ('SELL' if direction_code == -1 else 'FLAT')
    stop = target = None
    if direction == 'BUY':
        stop = round(price - atr_stop_mult * a, 5)
        target = round(price + atr_target_mult * a, 5)
    elif direction == 'SELL':
        stop = round(price + atr_stop_mult * a, 5)
        target = round(price - atr_target_mult * a, 5)
    return {
        'direction': direction,
        'price': round(price, 5),
        'stop': stop,
        'target': target,
        'atr': round(a, 5),
    }


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
def get_mt5_data(symbol, bars=500):
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No MT5 data for {symbol}. Check Market Watch symbol name for your broker."
        )

    df = pd.DataFrame(rates)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    return df[['date', 'open', 'high', 'low', 'close', 'tick_volume']]


def get_binance_data(symbol, interval='1d', bars=500):
    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={'symbol': symbol, 'interval': interval, 'limit': bars},
        timeout=12,
    )
    resp.raise_for_status()
    raw = resp.json()

    if not raw:
        raise RuntimeError(f"No Binance data for {symbol}")

    df = pd.DataFrame(raw, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore',
    ])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(float)
    df['tick_volume'] = df['volume'].astype(float)
    return df[['date', 'open', 'high', 'low', 'close', 'tick_volume']]


def _fmp_get_json(url, params):
    resp = requests.get(url, params=params, timeout=12)
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict) and payload.get('Error Message'):
        raise RuntimeError(payload['Error Message'])
    return payload


def get_fmp_data(symbol, bars=500, api_key=None):
    """
    Best-effort FMP daily OHLC loader.
    FMP has multiple endpoint shapes across asset classes/plans, so we try a
    short list and normalize whichever responds successfully.
    """
    key = api_key or os.environ.get('FMP_API_KEY', '')
    if not key:
        raise RuntimeError("FMP_API_KEY missing. Set env var or pass --fmp-api-key.")

    target = FMP_SYMBOL_MAP.get(symbol, symbol)
    endpoints = [
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{target}",
        f"https://financialmodelingprep.com/api/v3/historical-price-full/forex/{target}",
        f"https://financialmodelingprep.com/api/v3/historical-price-full/crypto/{target}",
    ]

    last_error = None
    for url in endpoints:
        try:
            payload = _fmp_get_json(url, {'apikey': key, 'serietype': 'line'})
            if isinstance(payload, dict) and isinstance(payload.get('historical'), list):
                hist = payload['historical'][:bars]
                if not hist:
                    continue

                df = pd.DataFrame(hist)
                required = {'date', 'open', 'high', 'low', 'close'}
                if not required.issubset(df.columns):
                    continue

                for col in ['open', 'high', 'low', 'close']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['tick_volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').dropna(subset=['open', 'high', 'low', 'close'])
                return df[['date', 'open', 'high', 'low', 'close', 'tick_volume']].tail(bars).reset_index(drop=True)
        except Exception as exc:
            last_error = str(exc)

    raise RuntimeError(f"FMP data unavailable for {symbol}. Last error: {last_error}")


def generate_synthetic_data(bars=600, start_price=100.0, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, bars)
    close = start_price * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.003, bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, bars)))
    open_ = np.roll(close, 1)
    open_[0] = start_price
    dates = pd.date_range(end=datetime.date.today(), periods=bars, freq='D')
    return pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'tick_volume': rng.integers(1000, 5000, bars),
    })


def _safe_div(a, b, default=0.0):
    return float(a / b) if b else float(default)


def _get_market_context(df):
    atr_value = float(atr(df).iloc[-1])
    close = float(df['close'].iloc[-1])
    atr_pct = _safe_div(atr_value, close)

    ema_fast = ema(df['close'], 20)
    ema_slow = ema(df['close'], 50)
    trend_slope = _safe_div(float(ema_fast.iloc[-1] - ema_slow.iloc[-1]), close)

    rsi_value = float(rsi(df['close'], 14).iloc[-1])
    upper, mid, lower = bollinger_bands(df['close'], 20, 2)
    bb_width = _safe_div(float((upper.iloc[-1] - lower.iloc[-1])), float(mid.iloc[-1]), default=0.0)

    if atr_pct < 0.01:
        vol_regime = 'low_vol'
    elif atr_pct < 0.03:
        vol_regime = 'mid_vol'
    else:
        vol_regime = 'high_vol'

    trend_regime = 'trending' if abs(trend_slope) >= 0.001 else 'choppy'
    if rsi_value < 35:
        mom_regime = 'oversold'
    elif rsi_value > 65:
        mom_regime = 'overbought'
    else:
        mom_regime = 'neutral'

    return {
        'atr_pct': round(atr_pct, 6),
        'trend_slope': round(trend_slope, 6),
        'rsi': round(rsi_value, 3),
        'bb_width': round(bb_width, 6),
        'vol_regime': vol_regime,
        'trend_regime': trend_regime,
        'mom_regime': mom_regime,
        'regime_key': f"{vol_regime}|{trend_regime}|{mom_regime}",
    }


def _within_metals_hours(now_dt, start_hour, end_hour):
    hour = now_dt.hour
    return start_hour <= hour < end_hour


def _is_market_moving(context, min_atr_pct=DEFAULT_MIN_ATR_PCT_MOVING):
    atr_pct = float(context.get('atr_pct', 0.0))
    bb_width = float(context.get('bb_width', 0.0))
    return atr_pct >= min_atr_pct and bb_width >= (min_atr_pct * 0.8)


def get_adaptive_confidence_floor(
    journal_path,
    target_precision=0.90,
    min_samples=40,
    lookback_rows=800,
    fallback=DEFAULT_OTHER_MIN_CONF,
):
    """
    Learns the minimum confidence that historically reached target precision.
    Precision here is: right / (right + wrong), evaluated on reviewed signals.
    """
    if not os.path.exists(journal_path):
        return fallback

    try:
        journal = pd.read_csv(journal_path)
    except Exception:
        return fallback

    if journal.empty:
        return fallback

    for col in ['outcome', 'confidence', 'evaluated']:
        if col not in journal.columns:
            return fallback

    hist = journal.tail(lookback_rows).copy()
    hist['outcome'] = hist['outcome'].fillna('').astype(str)
    hist['confidence'] = pd.to_numeric(hist['confidence'], errors='coerce')
    hist['evaluated'] = hist['evaluated'].fillna(False).astype(bool)

    hist = hist[(hist['evaluated']) & (hist['outcome'].isin(['right', 'wrong'])) & (hist['confidence'].notna())]
    if len(hist) < min_samples:
        return fallback

    best = None
    # Use a coarse, stable search over confidence thresholds.
    for th in np.arange(0.70, 0.991, 0.01):
        sub = hist[hist['confidence'] >= th]
        n = len(sub)
        if n < min_samples:
            continue
        precision = (sub['outcome'] == 'right').mean()
        if precision >= target_precision:
            best = float(th)
            break

    return round(best if best is not None else fallback, 3)


def load_learning_state(path=DEFAULT_LEARNING_STATE_PATH):
    if not os.path.exists(path):
        return {'symbol_strategy': {}, 'global_strategy': {}, 'regime_stats': {}, 'updated_at': None}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault('symbol_strategy', {})
        data.setdefault('global_strategy', {})
        data.setdefault('regime_stats', {})
        return data
    except Exception:
        return {'symbol_strategy': {}, 'global_strategy': {}, 'regime_stats': {}, 'updated_at': None}


def save_learning_state(state, path=DEFAULT_LEARNING_STATE_PATH):
    state = dict(state)
    state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _update_wl_entry(bucket, key, is_right):
    item = bucket.get(key, {'right': 0, 'wrong': 0})
    if is_right:
        item['right'] = int(item.get('right', 0)) + 1
    else:
        item['wrong'] = int(item.get('wrong', 0)) + 1
    bucket[key] = item


def _strategy_adjustment(symbol, strategy_name, learning_state):
    symbol_stats = learning_state.get('symbol_strategy', {}).get(symbol, {}).get(strategy_name, {})
    global_stats = learning_state.get('global_strategy', {}).get(strategy_name, {})

    s_total = int(symbol_stats.get('right', 0)) + int(symbol_stats.get('wrong', 0))
    g_total = int(global_stats.get('right', 0)) + int(global_stats.get('wrong', 0))

    s_adj = 0.0
    if s_total >= 5:
        s_wr = _safe_div(symbol_stats.get('right', 0), s_total, default=0.5)
        s_adj = (s_wr - 0.5) * 1.4

    g_adj = 0.0
    if g_total >= 8:
        g_wr = _safe_div(global_stats.get('right', 0), g_total, default=0.5)
        g_adj = (g_wr - 0.5) * 0.8

    return max(-0.8, min(0.8, s_adj + g_adj))


def _regime_adjustment(regime_key, learning_state):
    reg = learning_state.get('regime_stats', {}).get(regime_key, {})
    total = int(reg.get('right', 0)) + int(reg.get('wrong', 0))
    if total < 8:
        return 0.0
    wr = _safe_div(reg.get('right', 0), total, default=0.5)
    return max(-0.4, min(0.4, (wr - 0.5) * 0.8))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _get_data_for_symbol(
    symbol,
    bars,
    demo,
    use_mt5,
    crypto_source,
    metals_source,
    fmp_api_key,
):
    if demo:
        seed = sum(ord(c) for c in symbol) % 1000
        return generate_synthetic_data(bars=max(bars, 400), seed=seed)

    if symbol in CRYPTO:
        if crypto_source == 'binance':
            return get_binance_data(BINANCE_SYMBOL_MAP.get(symbol, symbol), bars=bars)
        if crypto_source == 'mt5':
            return get_mt5_data(symbol, bars=bars)
        if crypto_source == 'fmp':
            return get_fmp_data(symbol, bars=bars, api_key=fmp_api_key)
        raise RuntimeError(f"Unsupported crypto source: {crypto_source}")

    if symbol in METALS:
        if metals_source == 'mt5':
            return get_mt5_data(symbol, bars=bars)
        if metals_source == 'fmp':
            return get_fmp_data(symbol, bars=bars, api_key=fmp_api_key)
        raise RuntimeError(f"Unsupported metals source: {metals_source}")

    if use_mt5:
        return get_mt5_data(symbol, bars=bars)

    raise RuntimeError("No data source configured")


def generate_daily_signals(
    use_mt5=True,
    lookback=90,
    demo=False,
    crypto_source='binance',
    metals_source='mt5',
    folds=3,
    transaction_cost_bps=3.0,
    slippage_bps=1.0,
    min_confidence=0.55,
    fmp_api_key=None,
    learning_state=None,
    signal_date=None,
    now_dt=None,
    metals_start_hour=DEFAULT_METALS_START_HOUR,
    metals_end_hour=DEFAULT_METALS_END_HOUR,
    other_min_conf=DEFAULT_OTHER_MIN_CONF,
    other_max_conf=DEFAULT_OTHER_MAX_CONF,
    min_atr_pct_moving=DEFAULT_MIN_ATR_PCT_MOVING,
):
    learning_state = learning_state or {'symbol_strategy': {}, 'global_strategy': {}, 'regime_stats': {}}
    signal_date = signal_date or datetime.date.today()
    now_dt = now_dt or datetime.datetime.now()
    universe = get_today_universe()
    report = []

    for symbol in universe:
        try:
            ppy = 365 if symbol in CRYPTO else 252
            bars = max(lookback + 260, 420)
            df = _get_data_for_symbol(
                symbol=symbol,
                bars=bars,
                demo=demo,
                use_mt5=use_mt5,
                crypto_source=crypto_source,
                metals_source=metals_source,
                fmp_api_key=fmp_api_key,
            )

            ranked = rank_strategies(
                df,
                lookback=lookback,
                periods_per_year=ppy,
                folds=folds,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )

            context = _get_market_context(df)
            adjusted_ranked = []
            for strategy_name, metrics in ranked:
                m = dict(metrics)
                learn_adj = _strategy_adjustment(symbol, strategy_name, learning_state)
                regime_adj = _regime_adjustment(context['regime_key'], learning_state)
                m['learning_adjustment'] = round(learn_adj + regime_adj, 3)
                m['score_adjusted'] = round(m['score'] + m['learning_adjustment'], 3)
                adjusted_ranked.append((strategy_name, m))
            adjusted_ranked.sort(key=lambda kv: kv[1]['score_adjusted'], reverse=True)

            best_name, best_metrics = adjusted_ranked[0]
            second_score = (
                adjusted_ranked[1][1]['score_adjusted']
                if len(adjusted_ranked) > 1
                else adjusted_ranked[0][1]['score_adjusted']
            )
            confidence = _signal_confidence(
                best_score=best_metrics['score_adjusted'],
                second_score=second_score,
                best_sharpe=best_metrics['sharpe'],
                best_stability=best_metrics.get('stability', 0.0),
            )

            live = get_live_signal(df, best_name)
            raw_direction = live['direction']

            is_weekday = signal_date.weekday() < 5
            in_metals_window = _within_metals_hours(now_dt, metals_start_hour, metals_end_hour)
            market_moving = _is_market_moving(context, min_atr_pct=min_atr_pct_moving)

            # Rule 1: metals signals only during weekdays 04:00-18:00 local time.
            if symbol in METALS and is_weekday and not in_metals_window and live['direction'] != 'FLAT':
                live = dict(live)
                live['direction'] = 'FLAT'
                live['stop'] = None
                live['target'] = None

            # Rule 2: all other signals only when market is moving and confidence in [80%, 95%].
            if symbol not in METALS and live['direction'] != 'FLAT':
                if (not market_moving) or (confidence < other_min_conf) or (confidence > other_max_conf):
                    live = dict(live)
                    live['direction'] = 'FLAT'
                    live['stop'] = None
                    live['target'] = None

            if confidence < min_confidence and live['direction'] != 'FLAT':
                live = dict(live)
                live['direction'] = 'FLAT'
                live['stop'] = None
                live['target'] = None

            report.append({
                'symbol': symbol,
                'active_strategy': best_name,
                'strategy_metrics': best_metrics,
                'signal': live,
                'raw_direction': raw_direction,
                'confidence': confidence,
                'all_rankings': [(n, m['score_adjusted']) for n, m in adjusted_ranked],
                'source': crypto_source if symbol in CRYPTO else metals_source,
                'context': context,
                'signal_date': str(signal_date),
                'market_moving': market_moving,
                'in_metals_window': in_metals_window,
            })
        except Exception as exc:
            report.append({'symbol': symbol, 'error': str(exc)})

    return report


def append_signal_journal(report, path=DEFAULT_SIGNAL_JOURNAL_PATH):
    rows = []
    for r in report:
        if 'error' in r:
            continue
        rows.append({
            'signal_date': r.get('signal_date', str(datetime.date.today())),
            'symbol': r['symbol'],
            'source': r.get('source', ''),
            'strategy': r['active_strategy'],
            'direction': r['signal']['direction'],
            'raw_direction': r.get('raw_direction', r['signal']['direction']),
            'price': r['signal']['price'],
            'confidence': r.get('confidence', 0.0),
            'atr_pct': r.get('context', {}).get('atr_pct', np.nan),
            'trend_slope': r.get('context', {}).get('trend_slope', np.nan),
            'rsi': r.get('context', {}).get('rsi', np.nan),
            'bb_width': r.get('context', {}).get('bb_width', np.nan),
            'regime_key': r.get('context', {}).get('regime_key', ''),
            'evaluated': False,
            'outcome': '',
            'next_return': np.nan,
            'signed_return': np.nan,
            'evaluated_at': '',
            'notes': '',
        })

    if not rows:
        return 0

    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        # Keep only latest snapshot for each date-symbol pair when rerun same day.
        existing = existing[~existing.set_index(['signal_date', 'symbol']).index.isin(
            new_df.set_index(['signal_date', 'symbol']).index
        )]
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False)
    return len(new_df)


def _direction_to_sign(direction):
    if direction == 'BUY':
        return 1
    if direction == 'SELL':
        return -1
    return 0


def _get_next_bar_return(symbol, source, signal_date, demo, use_mt5, fmp_api_key):
    bars = 25
    if demo:
        seed = sum(ord(c) for c in symbol) % 1000
        df = generate_synthetic_data(bars=bars, seed=seed)
    else:
        if symbol in CRYPTO:
            if source == 'binance':
                df = get_binance_data(BINANCE_SYMBOL_MAP.get(symbol, symbol), bars=bars)
            elif source == 'fmp':
                df = get_fmp_data(symbol, bars=bars, api_key=fmp_api_key)
            else:
                df = get_mt5_data(symbol, bars=bars)
        else:
            if source == 'fmp':
                df = get_fmp_data(symbol, bars=bars, api_key=fmp_api_key)
            else:
                df = get_mt5_data(symbol, bars=bars)

    df = df.copy()
    df['day'] = pd.to_datetime(df['date']).dt.date
    idx = df.index[df['day'] == signal_date]
    if len(idx) == 0:
        return None
    i = int(idx[-1])
    if i + 1 >= len(df):
        return None

    c0 = float(df.loc[i, 'close'])
    c1 = float(df.loc[i + 1, 'close'])
    if c0 == 0:
        return None
    return (c1 / c0) - 1.0


def review_and_learn_signals(
    journal_path=DEFAULT_SIGNAL_JOURNAL_PATH,
    learning_state_path=DEFAULT_LEARNING_STATE_PATH,
    min_move_bps=0.0,
    demo=False,
    use_mt5=True,
    fmp_api_key=None,
):
    if not os.path.exists(journal_path):
        return {'reviewed': 0, 'right': 0, 'wrong': 0, 'pending': 0, 'message': 'No journal found yet.'}

    journal = pd.read_csv(journal_path)
    if journal.empty:
        return {'reviewed': 0, 'right': 0, 'wrong': 0, 'pending': 0, 'message': 'Journal is empty.'}

    # Normalize potentially empty columns that pandas can auto-read as float.
    for c in ['outcome', 'evaluated_at', 'notes', 'direction', 'raw_direction', 'strategy', 'source', 'regime_key']:
        if c in journal.columns:
            journal[c] = journal[c].fillna('').astype(str)

    for col in ['evaluated']:
        if col in journal.columns:
            journal[col] = journal[col].fillna(False).astype(bool)

    learning_state = load_learning_state(learning_state_path)
    pending_idx = journal.index[journal['evaluated'] == False]
    reviewed = right = wrong = source_errors = 0
    threshold = min_move_bps / 10000.0

    for idx in pending_idx:
        row = journal.loc[idx]
        signal_date = pd.to_datetime(row['signal_date']).date()
        if signal_date >= datetime.date.today():
            continue

        try:
            next_ret = _get_next_bar_return(
                symbol=row['symbol'],
                source=row.get('source', 'mt5'),
                signal_date=signal_date,
                demo=demo,
                use_mt5=use_mt5,
                fmp_api_key=fmp_api_key,
            )
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
            source_errors += 1
            if 'notes' in journal.columns:
                existing_raw = journal.at[idx, 'notes']
                existing_note = '' if pd.isna(existing_raw) else str(existing_raw).strip()
                err_note = (
                    f"review_fetch_error[{row['symbol']} {signal_date}]: {exc}"
                )
                journal.at[idx, 'notes'] = f"{existing_note}; {err_note}" if existing_note else err_note
            continue
        if next_ret is None:
            continue

        sign = _direction_to_sign(str(row.get('raw_direction', row.get('direction', 'FLAT'))))
        signed = sign * next_ret

        if sign == 0:
            outcome = 'flat'
        elif signed > threshold:
            outcome = 'right'
            right += 1
        else:
            outcome = 'wrong'
            wrong += 1

        journal.at[idx, 'evaluated'] = True
        journal.at[idx, 'outcome'] = outcome
        journal.at[idx, 'next_return'] = round(float(next_ret), 6)
        journal.at[idx, 'signed_return'] = round(float(signed), 6)
        journal.at[idx, 'evaluated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        reviewed += 1

        if outcome in ('right', 'wrong'):
            is_right = outcome == 'right'
            symbol_bucket = learning_state.setdefault('symbol_strategy', {}).setdefault(row['symbol'], {})
            _update_wl_entry(symbol_bucket, row['strategy'], is_right)

            global_bucket = learning_state.setdefault('global_strategy', {})
            _update_wl_entry(global_bucket, row['strategy'], is_right)

            regime_bucket = learning_state.setdefault('regime_stats', {})
            _update_wl_entry(regime_bucket, str(row.get('regime_key', 'unknown')), is_right)

    journal.to_csv(journal_path, index=False)
    save_learning_state(learning_state, learning_state_path)

    pending_after = int((journal['evaluated'] == False).sum())
    return {
        'reviewed': reviewed,
        'right': right,
        'wrong': wrong,
        'source_errors': source_errors,
        'pending': pending_after,
        'message': (
            f"Reviewed {reviewed}, right {right}, wrong {wrong}, "
            f"source errors {source_errors}, pending {pending_after}"
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report_text(report):
    label = "METALS (weekday)" if any(r.get('symbol') in METALS for r in report) else "CRYPTO (weekend)"
    lines = [f"Daily Signal Report - {label} - {datetime.date.today()}", ""]

    for r in report:
        if 'error' in r:
            lines.append(f"{r['symbol']}: ERROR - {r['error']}")
            continue

        s = r['signal']
        m = r['strategy_metrics']
        lines.append(
            f"{r['symbol']}: {s['direction']} @ {s['price']} | Stop {s['stop']} | Target {s['target']} | "
            f"Conf {r['confidence']:.2f}"
        )
        lines.append(
            f"   Strategy: {r['active_strategy']} ({r['source']}) | Sharpe {m['sharpe']} | PF {m['profit_factor']} | "
            f"WinRate {m['win_rate']:.0%} | DD {m['max_drawdown']:.1%} | Trades {m['trades']} | Stability {m.get('stability', 0):.2f}"
        )
        lines.append(f"   Rankings: {r['all_rankings']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def print_report(report):
    print("\n" + format_report_text(report))


def export_csv(report, path='daily_signals.csv'):
    rows = []
    for r in report:
        if 'error' in r:
            continue

        s, m = r['signal'], r['strategy_metrics']
        rows.append({
            'date': datetime.date.today(),
            'symbol': r['symbol'],
            'direction': s['direction'],
            'price': s['price'],
            'stop': s['stop'],
            'target': s['target'],
            'confidence': r.get('confidence'),
            'active_strategy': r['active_strategy'],
            'source': r.get('source'),
            'sharpe': m['sharpe'],
            'profit_factor': m['profit_factor'],
            'win_rate': m['win_rate'],
            'max_drawdown': m['max_drawdown'],
            'stability': m.get('stability', 0.0),
        })

    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Exported: {path}")


def email_report(report):
    import smtplib
    from email.mime.text import MIMEText

    cfg = {k: os.environ.get(v, '') for k, v in {
        'server': 'SIGNAL_SMTP_SERVER',
        'port': 'SIGNAL_SMTP_PORT',
        'user': 'SIGNAL_SMTP_USER',
        'password': 'SIGNAL_SMTP_PASSWORD',
        'from_addr': 'SIGNAL_FROM_EMAIL',
        'to_addr': 'SIGNAL_TO_EMAIL',
    }.items()}

    if not cfg['user'] or not cfg['password']:
        print("Email skipped: set SIGNAL_SMTP_* environment variables first.")
        return

    msg = MIMEText(format_report_text(report))
    msg['Subject'] = f"Daily Signal Report - {datetime.date.today()}"
    msg['From'], msg['To'] = cfg['from_addr'], cfg['to_addr']

    with smtplib.SMTP(cfg['server'], int(cfg['port'] or 587)) as server:
        server.starttls()
        server.login(cfg['user'], cfg['password'])
        server.sendmail(cfg['from_addr'], [cfg['to_addr']], msg.as_string())

    print(f"Emailed to {cfg['to_addr']}")


def telegram_report(report):
    token = os.environ.get('SIGNAL_TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.environ.get('SIGNAL_TELEGRAM_CHAT_ID', '').strip()

    if not token or not chat_id:
        print("Telegram skipped: set SIGNAL_TELEGRAM_BOT_TOKEN and SIGNAL_TELEGRAM_CHAT_ID.")
        return

    text = format_report_text(report)
    if len(text) > 3800:
        text = text[:3790] + "\n\n...truncated"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        data={
            'chat_id': chat_id,
            'text': text,
            'disable_web_page_preview': True,
        },
        timeout=12,
    )
    resp.raise_for_status()
    print(f"Telegram sent to chat {chat_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-strategy daily signal generator")
    parser.add_argument('--demo', action='store_true', help='Use synthetic data and test full pipeline')
    parser.add_argument('--lookback', type=int, default=90, help='Rolling ranking window in bars')
    parser.add_argument('--folds', type=int, default=3, help='Walk-forward folds for ranking stability')
    parser.add_argument('--eod-review', action='store_true', help='Evaluate pending past signals and update learning state')
    parser.add_argument('--review-only', action='store_true', help='Run end-of-day review and exit')
    parser.add_argument('--review-min-move-bps', type=float, default=0.0, help='Move threshold in bps to score a signal as right')
    parser.add_argument('--journal-path', type=str, default=DEFAULT_SIGNAL_JOURNAL_PATH, help='Path for signal journal CSV')
    parser.add_argument('--learning-state-path', type=str, default=DEFAULT_LEARNING_STATE_PATH, help='Path for learning state JSON')
    parser.add_argument('--email', action='store_true', help='Email report via SIGNAL_SMTP_* env vars')
    parser.add_argument('--telegram', action='store_true', help='Send Telegram alert via SIGNAL_TELEGRAM_* env vars')
    parser.add_argument('--crypto-source', choices=['binance', 'mt5', 'fmp'], default='binance', help='Crypto data source')
    parser.add_argument('--metals-source', choices=['mt5', 'fmp'], default='mt5', help='Metals data source')
    parser.add_argument('--metals-start-hour', type=int, default=DEFAULT_METALS_START_HOUR, help='Local start hour for weekday metals signals')
    parser.add_argument('--metals-end-hour', type=int, default=DEFAULT_METALS_END_HOUR, help='Local end hour for weekday metals signals')
    parser.add_argument('--other-min-confidence', type=float, default=DEFAULT_OTHER_MIN_CONF, help='Min confidence for non-metals signal generation')
    parser.add_argument('--other-max-confidence', type=float, default=DEFAULT_OTHER_MAX_CONF, help='Max confidence for non-metals signal generation')
    parser.add_argument('--adaptive-confidence', action='store_true', help='Learn confidence floor from historical reviewed outcomes')
    parser.add_argument('--target-precision', type=float, default=0.90, help='Target precision for adaptive-confidence mode (0.0-1.0)')
    parser.add_argument('--adaptive-min-samples', type=int, default=40, help='Minimum reviewed samples required for adaptive-confidence mode')
    parser.add_argument('--timezone', type=str, default='UTC', help='IANA timezone for market-hour gating, e.g. America/New_York')
    parser.add_argument('--min-atr-pct-moving', type=float, default=DEFAULT_MIN_ATR_PCT_MOVING, help='ATR%% threshold used to treat market as moving')
    parser.add_argument('--transaction-cost-bps', type=float, default=3.0, help='Round-trip transaction cost in bps')
    parser.add_argument('--slippage-bps', type=float, default=1.0, help='Estimated slippage in bps')
    parser.add_argument('--min-confidence', type=float, default=0.55, help='Min confidence before non-FLAT signal')
    parser.add_argument('--fmp-api-key', type=str, default='', help='Optional FMP API key (otherwise uses FMP_API_KEY env)')
    args = parser.parse_args()

    if args.eod_review:
        review_summary = review_and_learn_signals(
            journal_path=args.journal_path,
            learning_state_path=args.learning_state_path,
            min_move_bps=args.review_min_move_bps,
            demo=args.demo,
            use_mt5=not args.demo,
            fmp_api_key=args.fmp_api_key or None,
        )
        print(f"EOD review: {review_summary['message']}")
        if args.review_only:
            raise SystemExit(0)

    learning_state = load_learning_state(args.learning_state_path)

    try:
        now_local = datetime.datetime.now(ZoneInfo(args.timezone))
    except Exception as exc:
        raise SystemExit(f"Invalid timezone '{args.timezone}': {exc}")

    effective_other_min_conf = args.other_min_confidence
    if args.adaptive_confidence:
        effective_other_min_conf = get_adaptive_confidence_floor(
            journal_path=args.journal_path,
            target_precision=max(0.0, min(1.0, args.target_precision)),
            min_samples=max(10, int(args.adaptive_min_samples)),
            fallback=args.other_min_confidence,
        )
        print(
            f"Adaptive confidence floor active: {effective_other_min_conf:.2f} "
            f"(target precision {args.target_precision:.2f}, min samples {args.adaptive_min_samples})"
        )

    report = generate_daily_signals(
        use_mt5=not args.demo,
        lookback=args.lookback,
        demo=args.demo,
        crypto_source=args.crypto_source,
        metals_source=args.metals_source,
        folds=args.folds,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        min_confidence=args.min_confidence,
        fmp_api_key=args.fmp_api_key or None,
        learning_state=learning_state,
        signal_date=datetime.date.today(),
        now_dt=now_local,
        metals_start_hour=args.metals_start_hour,
        metals_end_hour=args.metals_end_hour,
        other_min_conf=effective_other_min_conf,
        other_max_conf=args.other_max_confidence,
        min_atr_pct_moving=args.min_atr_pct_moving,
    )

    print_report(report)
    export_csv(report, path='daily_signals_demo.csv' if args.demo else 'daily_signals.csv')
    written = append_signal_journal(report, path=args.journal_path)
    print(f"Journal updated: {written} signal rows appended to {args.journal_path}")

    if args.email:
        email_report(report)
    if args.telegram:
        telegram_report(report)
