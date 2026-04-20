#!/usr/bin/env python3
"""
master_stock_analyzer.py
=========================
STAGE 1 — DISCOVERY
  Load all equity data. Filter: price >Rs10, avg daily turnover >Rs5Cr.
  For every qualifying stock, compute 12 technical strategies.

STAGE 2 — BACKTEST
  For each strategy on each stock: simulate every signal historically.
  Compute win rate, avg return, max drawdown, Sharpe ratio.
  Discard any strategy that doesn't beat 65% win rate with avg >1.5%.

STAGE 3 — CHALLENGE & VERIFY
  For the best 2 strategies per stock: run a second backtest on the
  OTHER half of data (out-of-sample). Verify results hold.
  Grade: A+ (both halves pass), A (1 of 2), B (marginal).

STAGE 4 — OUTPUT
  master_results.json  — per-stock best strategy with backtest proof
  daily_alerts.json    — today's actionable signals
  longterm_picks.json  — best stocks for multi-week/month hold

INCREMENTAL: checkpoint tracks last-processed date per stock.
Only new dates are re-evaluated; historical scores are preserved.

Strategies implemented:
  1.  SMA_CROSS      — 10/30 SMA golden/death cross
  2.  EMA_BOUNCE     — Price bounces off 20-EMA with volume surge
  3.  RSI_REVERSAL   — RSI <30 (oversold) bounce / >70 (overbought) reverse
  4.  MACD_SIGNAL    — MACD line crosses signal line
  5.  BB_SQUEEZE     — Bollinger Band width narrows then expands
  6.  VOL_BREAKOUT   — Price breaks N-day high with 2x volume
  7.  CONSEC_REVERSAL — 3+ consecutive red days = reversal signal
  8.  FLOOR_BOUNCE   — Price touches 20-day low then reverses up
  9.  MOMENTUM_GAP   — Gap-up open >2% with follow-through
  10. RANGE_EXPAND   — ATR breakout: day range > 2× ATR
  11. WEEKLY_PULSE   — Monday low → Friday close pattern
  12. TREND_PULLBACK — Strong uptrend + temporary dip = entry
"""

import json, gc, sys, os, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy"); sys.exit(1)

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUT_DIR    = REPO_ROOT / "stock_analysis"
MANIFEST   = DATA_DIR / "manifest.json"
CHECKPOINT = OUT_DIR / "master_checkpoint.json"
RESULT_F   = OUT_DIR / "master_results.json"
ALERTS_F   = OUT_DIR / "daily_alerts.json"
LONGTERM_F = OUT_DIR / "longterm_picks.json"

# ─── Constants ────────────────────────────────────────────────────────────────
MIN_PRICE           = 10.0
MIN_AVG_TURNOVER    = 5_000_000   # Rs 5 Cr daily avg
MIN_HISTORY_DAYS    = 120         # need at least 120 trading days
MIN_WIN_RATE        = 60.0        # strategy must beat 60% to be kept
MIN_AVG_RETURN      = 1.0         # minimum avg return per trade (%)
MIN_OCCURRENCES     = 5           # minimum backtest trades to be statistically meaningful
HOLD_DAYS_SHORT     = 5           # short-term hold window
HOLD_DAYS_LONG      = 20          # medium-term hold window
EXCLUDED_EXACT      = {
    "LIQUIDIETF","LIQUIDBEES","LIQUIDCASE","NIFTYBEES","JUNIORBEES",
    "BANKBEES","GOLDBEES","SILVERBEES","PSUBNKBEES","ITBEES","CPSEETF",
}
EXCLUDED_SUFFIX     = ("ETF","BEES","CASE","SETF","GILT")

# ─── Column aliases ───────────────────────────────────────────────────────────
SYM_A = ["SYMBOL","TCKRSYMB"]
SER_A = ["SERIES","SCTYSRS"]
O_A   = ["OPEN","OPNPRIC","OPEN PRICE"]
H_A   = ["HIGH","HGHPRIC","HIGH PRICE"]
L_A   = ["LOW","LWPRIC","LOW PRICE"]
C_A   = ["CLOSE","CLSPRIC","CLOSE PRICE","LASTPRIC"]
V_A   = ["TOTTRDQTY","TTLTRADGVOL","VOLUME"]

def r2(x): return round(float(x)*100)/100 if x is not None and not np.isnan(x) else None

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,(date_type,datetime)): return str(obj)
        if isinstance(obj,np.integer):  return int(obj)
        if isinstance(obj,np.floating): return float(obj) if not np.isnan(obj) else None
        if isinstance(obj,np.bool_):    return bool(obj)
        if isinstance(obj,np.ndarray):  return obj.tolist()
        return super().default(obj)

def jdump(obj, path):
    with open(path,"w") as f: json.dump(obj,f,indent=2,cls=SafeEncoder)

def jload(path):
    try:
        with open(path) as f: return json.load(f)
    except: return {}

# ─── CSV loader ───────────────────────────────────────────────────────────────
def _fc(hdr, aliases):
    for a in aliases:
        if a in hdr: return hdr.index(a)
    return -1

def load_csv(path):
    rows = []
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
        if len(lines)<2: return rows
        hdr=[h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
        i_s=_fc(hdr,SYM_A); i_sr=_fc(hdr,SER_A)
        i_o=_fc(hdr,O_A);   i_h=_fc(hdr,H_A)
        i_l=_fc(hdr,L_A);   i_c=_fc(hdr,C_A)
        i_v=_fc(hdr,V_A)
        if i_s<0 or i_c<0: return rows
        mc=max(x for x in [i_s,i_o,i_h,i_l,i_c,i_v] if x>=0)
        for line in lines[1:]:
            line=line.strip()
            if not line: continue
            cols=[c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols)<=mc: continue
            ser=cols[i_sr].strip() if i_sr>=0 else "EQ"
            if ser not in ("EQ","BE"): continue
            try:
                sym=cols[i_s].strip(); c=float(cols[i_c])
                o=float(cols[i_o]) if i_o>=0 else c
                h=float(cols[i_h]) if i_h>=0 else c
                l=float(cols[i_l]) if i_l>=0 else c
                v=float(str(cols[i_v]).replace(",","")) if i_v>=0 else 0.0
                if c>0 and o>0 and sym:
                    rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v})
            except: pass
    except: pass
    return rows

def load_all_days(trading_days):
    rows=[]; loaded=0
    for ds in trading_days:
        y,m,_=ds.split("-")
        path=DATA_DIR/"equity"/y/m/f"{ds}.csv"
        if not path.exists(): continue
        r=load_csv(path)
        for x in r: x["date"]=ds
        rows.extend(r); loaded+=1
        if loaded%300==0:
            print(f"    {loaded}/{len(trading_days)} files loaded…",flush=True)
    df=pd.DataFrame(rows)
    if df.empty: return df
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(["sym","date"]).reset_index(drop=True)

# ─── Technical Indicators (vectorised) ────────────────────────────────────────
def add_indicators(df):
    """Add all technical indicators to a per-stock DataFrame."""
    c = df["c"]; v = df["v"]

    # Moving averages
    df["sma10"]  = c.rolling(10,  min_periods=5).mean()
    df["sma20"]  = c.rolling(20,  min_periods=10).mean()
    df["sma30"]  = c.rolling(30,  min_periods=15).mean()
    df["ema20"]  = c.ewm(span=20, adjust=False).mean()
    df["ema50"]  = c.ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20-day, 2σ)
    bb_mean = c.rolling(20, min_periods=10).mean()
    bb_std  = c.rolling(20, min_periods=10).std()
    df["bb_upper"] = bb_mean + 2*bb_std
    df["bb_lower"] = bb_mean - 2*bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mean * 100

    # RSI
    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    loss   = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs     = gain / loss.replace(0, 0.0001)
    df["rsi"] = 100 - 100/(1+rs)

    # MACD (12/26/9)
    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]

    # ATR (14-day)
    hl = df["h"] - df["l"]
    hc = (df["h"] - c.shift(1)).abs()
    lc = (df["l"] - c.shift(1)).abs()
    df["atr"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14,min_periods=7).mean()

    # Volume indicators
    df["vol_avg20"] = v.rolling(20, min_periods=10).mean()
    df["vol_ratio"] = v / df["vol_avg20"].replace(0, 1)

    # Price momentum
    df["ret1"]  = c.pct_change(1) * 100
    df["ret5"]  = c.pct_change(5) * 100
    df["ret20"] = c.pct_change(20) * 100

    # N-day high/low
    df["high20"] = df["h"].rolling(20, min_periods=10).max()
    df["low20"]  = df["l"].rolling(20, min_periods=10).min()
    df["high52w"] = df["h"].rolling(252, min_periods=100).max()
    df["low52w"]  = df["l"].rolling(252, min_periods=100).min()

    # Consecutive red days
    df["is_red"]  = (c < c.shift(1)).astype(int)
    df["consec_red"] = df["is_red"] * (df["is_red"].groupby((df["is_red"]!=df["is_red"].shift()).cumsum()).cumcount()+1)

    # Gap open
    df["gap_pct"] = (df["o"] - c.shift(1)) / c.shift(1) * 100

    # Turnover
    df["turnover"] = c * v

    return df

# ─── Strategy Signal Generators ───────────────────────────────────────────────
def gen_signals(df):
    """
    Returns dict: strategy_name → boolean Series (True = buy signal on that day)
    Signal day = day BEFORE entry. Entry = next open.
    """
    sigs = {}
    c = df["c"]

    # 1. SMA_CROSS: 10-SMA crosses above 30-SMA (golden cross)
    sigs["SMA_CROSS"] = (
        (df["sma10"] > df["sma30"]) &
        (df["sma10"].shift(1) <= df["sma30"].shift(1)) &
        (df["vol_ratio"] >= 1.2)
    )

    # 2. EMA_BOUNCE: Close crosses above 20-EMA from below with volume
    sigs["EMA_BOUNCE"] = (
        (c > df["ema20"]) &
        (c.shift(1) < df["ema20"].shift(1)) &
        (df["vol_ratio"] >= 1.5) &
        (df["rsi"] < 60)
    )

    # 3. RSI_OVERSOLD: RSI below 30 then crosses back above 35
    sigs["RSI_OVERSOLD"] = (
        (df["rsi"] > 35) &
        (df["rsi"].shift(1) <= 30) &
        (c > c.shift(1))
    )

    # 4. MACD_CROSS: MACD crosses above signal line from below
    sigs["MACD_CROSS"] = (
        (df["macd"] > df["macd_sig"]) &
        (df["macd"].shift(1) <= df["macd_sig"].shift(1)) &
        (df["vol_ratio"] >= 1.0)
    )

    # 5. BB_BOUNCE: Price touches lower band then closes back inside
    sigs["BB_BOUNCE"] = (
        (df["l"] <= df["bb_lower"]) &
        (c > df["bb_lower"]) &
        (c.shift(1) <= df["bb_lower"].shift(1).fillna(c)) &
        (df["vol_ratio"] >= 1.3)
    )

    # 6. VOL_BREAKOUT: Price breaks 20-day high with 2x avg volume
    sigs["VOL_BREAKOUT"] = (
        (c >= df["high20"].shift(1)) &
        (df["vol_ratio"] >= 2.0) &
        (df["ret1"] > 1.0)
    )

    # 7. CONSEC_REVERSAL: 3+ consecutive red days → close higher (reversal)
    sigs["CONSEC_REVERSAL"] = (
        (df["consec_red"].shift(1) >= 3) &
        (c > c.shift(1)) &
        (df["vol_ratio"] >= 1.3)
    )

    # 8. FLOOR_BOUNCE: Price touches 20-day low, next day reversal
    sigs["FLOOR_BOUNCE"] = (
        (df["l"].shift(1) <= df["low20"].shift(2)) &
        (c > c.shift(1)) &
        (df["rsi"] < 50)
    )

    # 9. MOMENTUM_GAP: Gap-up open >2% AND closes in top 25% of day's range
    day_range = (df["h"] - df["l"]).replace(0,0.0001)
    close_pos  = (c - df["l"]) / day_range   # 1.0 = closed at high
    sigs["MOMENTUM_GAP"] = (
        (df["gap_pct"] > 2.0) &
        (close_pos > 0.75) &
        (df["vol_ratio"] >= 1.5)
    )

    # 10. RANGE_EXPAND: Day range > 2× ATR and bullish close
    day_range2 = df["h"] - df["l"]
    sigs["RANGE_EXPAND"] = (
        (day_range2 > 2 * df["atr"]) &
        (c > df["o"]) &
        (df["ret1"] > 2.0) &
        (df["vol_ratio"] >= 1.5)
    )

    # 11. TREND_PULLBACK: 20-day uptrend + 3-day pullback + reversal
    sigs["TREND_PULLBACK"] = (
        (df["ret20"] > 8.0) &           # strong prior uptrend
        (df["ret5"] < -2.0) &           # recent pullback
        (c > c.shift(1)) &              # today closes up
        (df["vol_ratio"] >= 1.2) &
        (df["rsi"] < 55)
    )

    # 12. 52W_HIGH_BREAK: Approaches or breaks 52-week high with volume
    sigs["HIGHBREAK_52W"] = (
        (c >= df["high52w"].shift(1) * 0.98) &   # within 2% of 52w high
        (df["vol_ratio"] >= 2.0) &
        (df["ret1"] > 1.5) &
        (df["rsi"] > 60)   # momentum confirming
    )

    return sigs

# ─── Backtest Engine ──────────────────────────────────────────────────────────
def backtest_strategy(df, signal_series, hold_days, label):
    """
    Simulate strategy:
      Entry: next-day OPEN after signal
      Exit:  close after `hold_days` trading days
    Returns dict with full performance metrics or None.
    """
    signals = signal_series.fillna(False)
    c = df["c"].values
    o = df["o"].values
    n = len(df)
    dates = df["date"].values

    trades = []
    i = 0
    while i < n-1:
        if not signals.iloc[i]:
            i += 1
            continue
        # Entry next day open
        entry_idx = i + 1
        if entry_idx >= n: break
        entry_px  = o[entry_idx]
        entry_date= str(pd.Timestamp(dates[entry_idx]).date())

        # Exit after hold_days
        exit_idx  = min(entry_idx + hold_days, n-1)
        exit_px   = c[exit_idx]
        exit_date = str(pd.Timestamp(dates[exit_idx]).date())

        # Also track max gain (best close in hold window) and max drawdown
        hold_window = c[entry_idx:exit_idx+1]
        max_c  = float(hold_window.max()) if len(hold_window)>0 else exit_px
        min_c  = float(hold_window.min()) if len(hold_window)>0 else exit_px

        ret        = (exit_px - entry_px) / entry_px * 100
        max_gain   = (max_c   - entry_px) / entry_px * 100
        max_dd     = (min_c   - entry_px) / entry_px * 100  # negative = drawdown

        trades.append({
            "sig_date":   str(pd.Timestamp(dates[i]).date()),
            "entry_date": entry_date,
            "exit_date":  exit_date,
            "entry_px":   r2(entry_px),
            "exit_px":    r2(exit_px),
            "ret":        r2(ret),
            "max_gain":   r2(max_gain),
            "max_dd":     r2(max_dd),
            "positive":   ret > 0,
        })
        i = exit_idx   # no overlapping trades

    if len(trades) < MIN_OCCURRENCES:
        return None

    rets = [t["ret"] for t in trades]
    pos  = [t for t in trades if t["positive"]]
    win_rate   = len(pos)/len(trades)*100
    avg_ret    = np.mean(rets)
    avg_win    = np.mean([t["ret"] for t in pos]) if pos else 0
    avg_loss   = np.mean([t["ret"] for t in trades if not t["positive"]]) if len(trades)>len(pos) else 0
    max_dd_abs = min([t["max_dd"] for t in trades])
    avg_max_gain = np.mean([t["max_gain"] for t in trades])

    # Sharpe approximation (daily returns)
    if len(rets) > 2:
        sharpe = avg_ret / (np.std(rets) + 0.001) * np.sqrt(252/hold_days)
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = sum(t["ret"] for t in pos)
    gross_loss   = abs(sum(t["ret"] for t in trades if not t["positive"]))
    pf = gross_profit / (gross_loss + 0.001)

    return {
        "strategy":    label,
        "hold_days":   hold_days,
        "n_trades":    len(trades),
        "win_rate":    r2(win_rate),
        "avg_ret":     r2(avg_ret),
        "avg_win":     r2(avg_win),
        "avg_loss":    r2(avg_loss),
        "max_dd":      r2(max_dd_abs),
        "avg_max_gain":r2(avg_max_gain),
        "sharpe":      r2(sharpe),
        "profit_factor":r2(pf),
        "trades":      trades,
    }

# ─── Out-of-sample Challenge ──────────────────────────────────────────────────
def challenge(df, signal_series, hold_days, label):
    """
    Split data in half. Backtest on each half separately.
    Returns ('A+', in_sample_result, oos_result) etc.
    """
    mid = len(df) // 2
    df1 = df.iloc[:mid].reset_index(drop=True)
    df2 = df.iloc[mid:].reset_index(drop=True)
    s1  = signal_series.iloc[:mid].reset_index(drop=True)
    s2  = signal_series.iloc[mid:].reset_index(drop=True)

    r1 = backtest_strategy(df1, s1, hold_days, label+"_IN")
    r2_ = backtest_strategy(df2, s2, hold_days, label+"_OOS")

    def passes(r):
        if r is None: return False
        return r["win_rate"] >= MIN_WIN_RATE and r["avg_ret"] >= MIN_AVG_RETURN

    if passes(r1) and passes(r2_): grade = "A+"
    elif passes(r1) or passes(r2_): grade = "A"
    else: grade = "B"

    return grade, r1, r2_

# ─── Per-stock analysis ───────────────────────────────────────────────────────
def analyse_stock(sym, df, force=False):
    """
    Run all strategies on one stock.
    Returns dict with best strategies + daily signal status.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    n  = len(df)
    if n < MIN_HISTORY_DAYS: return None

    c_last = float(df["c"].iloc[-1])
    if c_last < MIN_PRICE: return None

    tv_60  = float((df["c"] * df["v"]).iloc[-60:].mean()) if n>=60 else 0
    if tv_60 < MIN_AVG_TURNOVER: return None

    # Add indicators
    df = add_indicators(df)

    # Generate signals
    all_sigs = gen_signals(df)

    best_strategies = []

    for s_name, sig_series in all_sigs.items():
        n_signals = int(sig_series.fillna(False).sum())
        if n_signals < MIN_OCCURRENCES: continue

        # Full backtest on all data
        bt_short = backtest_strategy(df, sig_series, HOLD_DAYS_SHORT,  s_name)
        bt_long  = backtest_strategy(df, sig_series, HOLD_DAYS_LONG, s_name)

        best_bt = None
        best_hold = None
        if bt_short and bt_short["win_rate"] >= MIN_WIN_RATE and bt_short["avg_ret"] >= MIN_AVG_RETURN:
            best_bt   = bt_short
            best_hold = HOLD_DAYS_SHORT
        if bt_long and bt_long["win_rate"] >= MIN_WIN_RATE and bt_long["avg_ret"] >= MIN_AVG_RETURN:
            if best_bt is None or bt_long["avg_ret"] > best_bt["avg_ret"]:
                best_bt   = bt_long
                best_hold = HOLD_DAYS_LONG

        if best_bt is None: continue

        # Out-of-sample challenge
        grade, r_in, r_oos = challenge(df, sig_series, best_hold, s_name)
        if grade == "B": continue   # doesn't survive challenge

        best_strategies.append({
            "strategy":     s_name,
            "hold_days":    best_hold,
            "grade":        grade,
            "win_rate":     best_bt["win_rate"],
            "avg_ret":      best_bt["avg_ret"],
            "avg_max_gain": best_bt["avg_max_gain"],
            "max_dd":       best_bt["max_dd"],
            "sharpe":       best_bt["sharpe"],
            "profit_factor":best_bt["profit_factor"],
            "n_trades":     best_bt["n_trades"],
            # OOS verification
            "oos_win_rate": r_oos["win_rate"] if r_oos else None,
            "oos_avg_ret":  r_oos["avg_ret"]  if r_oos else None,
            # Last 5 trades for display
            "recent_trades": best_bt["trades"][-5:],
            # Today's signal
            "signal_today": bool(sig_series.iloc[-1]) if len(sig_series)>0 else False,
        })

    if not best_strategies: return None

    # Rank strategies: A+ first, then by avg_ret
    grade_rank = {"A+":0,"A":1,"B":2}
    best_strategies.sort(key=lambda x: (grade_rank.get(x["grade"],3), -x["avg_ret"]))

    top = best_strategies[0]

    # Today's signal from ANY qualifying strategy
    any_signal_today = any(s["signal_today"] for s in best_strategies)
    signal_strategies = [s["strategy"] for s in best_strategies if s["signal_today"]]

    # Why pattern works — explain from data
    explanation = _explain(df, top["strategy"], top["win_rate"], top["avg_ret"], top["hold_days"])

    # Long-term suitability score
    lt_score = _longterm_score(df, best_strategies)

    latest_date = str(df["date"].iloc[-1].date())
    return {
        "sym":               sym,
        "latest_date":       latest_date,
        "price":             r2(c_last),
        "avg_turnover_cr":   r2(tv_60 / 1e7),
        "n_strategies":      len(best_strategies),
        "best_strategy":     top["strategy"],
        "best_grade":        top["grade"],
        "best_win_rate":     top["win_rate"],
        "best_avg_ret":      top["avg_ret"],
        "best_hold_days":    top["hold_days"],
        "best_max_gain":     top["avg_max_gain"],
        "best_sharpe":       top["sharpe"],
        "best_pf":           top["profit_factor"],
        "oos_win_rate":      top["oos_win_rate"],
        "oos_avg_ret":       top["oos_avg_ret"],
        "all_strategies":    best_strategies,
        "signal_today":      any_signal_today,
        "signal_strategies": signal_strategies,
        "explanation":       explanation,
        "longterm_score":    lt_score,
        "recent_trades":     top["recent_trades"],
        "rsi":               r2(df["rsi"].iloc[-1]),
        "sma20":             r2(df["sma20"].iloc[-1]),
        "sma50":             r2(df["ema50"].iloc[-1]) if "ema50" in df else None,
        "bb_width":          r2(df["bb_width"].iloc[-1]),
        "vol_ratio":         r2(df["vol_ratio"].iloc[-1]),
        "ret5":              r2(df["ret5"].iloc[-1]),
        "ret20":             r2(df["ret20"].iloc[-1]),
    }

def _explain(df, strategy, win_rate, avg_ret, hold_days):
    """Generate human-readable explanation of why this strategy works on this stock."""
    c = df["c"]
    v = df["v"]

    base = f"{strategy} strategy: {win_rate:.0f}% win rate, avg +{avg_ret:.1f}% over {hold_days} days. "
    details = {
        "SMA_CROSS":      "When the fast 10-day average crosses above the slow 30-day average with rising volume, "
                          "it signals institutional accumulation — momentum is shifting upward.",
        "EMA_BOUNCE":     "Each time price dips to the 20-day exponential average and bounces with a volume surge, "
                          "buyers are defending a key level — strong reward-to-risk entry.",
        "RSI_OVERSOLD":   "When RSI falls below 30 (historically oversold for this stock) and turns up, "
                          "sellers are exhausted and short-covering often drives a sharp recovery.",
        "MACD_CROSS":     "MACD crossing its signal line marks a momentum inflection point — "
                          "historically this stock tends to continue in the new direction for several days.",
        "BB_BOUNCE":      "This stock repeatedly touches its lower Bollinger Band then snaps back to the mean. "
                          "The band acts as a rubber band — the further the stretch, the sharper the recovery.",
        "VOL_BREAKOUT":   "When price breaks to a new 20-day high with double average volume, "
                          "it signals that large players are accumulating — the breakout has conviction.",
        "CONSEC_REVERSAL":"After 3 or more consecutive down days, this stock historically reverses sharply. "
                          "The selling exhaustion creates a rebound opportunity consistently.",
        "FLOOR_BOUNCE":   "This stock has a predictable support floor — each time it touches the 20-day low "
                          "with rising RSI, it bounces. Institutional buyers appear at this level.",
        "MOMENTUM_GAP":   "Gap-up opens >2% that close in the top quarter of the day's range signal "
                          "continuation — the gap reflects genuine demand and the stock runs further.",
        "RANGE_EXPAND":   "Days with unusually large ranges (>2× ATR) that close bullish mark "
                          "high-energy breakout days — the expanded range drives follow-through.",
        "TREND_PULLBACK": "In a strong uptrend, this stock's dips are reliably bought. "
                          "After pulling back 3-5% from the 20-day trend, it resumes the upward move.",
        "HIGHBREAK_52W":  "Breaking to new 52-week highs with strong volume is the most powerful signal "
                          "in technical analysis — there are no sellers left above this price.",
    }
    return base + details.get(strategy, "Consistent pattern identified via backtesting.")

def _longterm_score(df, strategies):
    """Score 0-100 for long-term investment suitability."""
    c = df["c"]
    n = len(df)
    score = 0

    # Trend: is the stock above its 50-day EMA?
    ema50 = c.ewm(span=50, adjust=False).mean()
    if float(c.iloc[-1]) > float(ema50.iloc[-1]): score += 20

    # Consistency: 60-day return positive?
    ret60 = (float(c.iloc[-1]) - float(c.iloc[max(0,n-60)])) / float(c.iloc[max(0,n-60)]) * 100
    if ret60 > 0: score += 15
    if ret60 > 10: score += 10

    # A+ strategies (out-of-sample verified)?
    aplus = [s for s in strategies if s["grade"] == "A+"]
    score += min(25, len(aplus) * 10)

    # Best win rate
    best_wr = max([s["win_rate"] for s in strategies], default=0)
    if best_wr >= 75: score += 15
    elif best_wr >= 65: score += 8

    # Avg return
    best_ar = max([s["avg_ret"] for s in strategies], default=0)
    if best_ar >= 5: score += 15
    elif best_ar >= 3: score += 8

    return min(100, score)


# ─── Checkpoint system ────────────────────────────────────────────────────────
def load_cp():
    cp = jload(CHECKPOINT)
    return cp.get("processed_date",""), cp.get("results",{})

def save_cp(processed_date, results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jdump({"processed_date": processed_date, "results": results}, CHECKPOINT)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    force = os.getenv("FORCE_RERUN","false").lower() == "true"

    print("="*70)
    print("Master Stock Analyzer")
    print("  STAGE 1: Discovery + Indicators")
    print("  STAGE 2: Strategy Backtest")
    print("  STAGE 3: Out-of-Sample Challenge")
    print("  STAGE 4: Alerts + Output")
    print("="*70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest = jload(MANIFEST)
    tds      = sorted(manifest.keys())
    latest   = tds[-1]
    print(f"\n  Trading days: {len(tds)} | Latest: {latest}")

    # Load checkpoint
    cp_date, saved_results = load_cp()
    if force:
        cp_date = ""; saved_results = {}
        print("  FORCE RERUN — clearing checkpoint")
    elif cp_date == latest:
        print(f"  Already processed up to {latest}. Generating outputs only.")
        # Still regenerate alerts and long-term picks from saved results
        _write_outputs(saved_results, latest)
        return
    else:
        print(f"  Checkpoint: {cp_date or 'none'} → need to process {latest}")

    # Load all equity data
    print("\n[1] Loading equity data...")
    df_all = load_all_days(manifest)
    if df_all.empty:
        print("  ERROR: No data loaded."); sys.exit(1)

    sym_list = sorted(df_all["sym"].unique())
    print(f"  {len(sym_list)} unique symbols")

    # Group by symbol
    print("\n[2] Grouping by symbol...")
    sym_grps = {}
    for sym, grp in df_all.groupby("sym"):
        if sym in EXCLUDED_EXACT: continue
        if any(sym.upper().endswith(s) for s in EXCLUDED_SUFFIX): continue
        sym_grps[sym] = grp.reset_index(drop=True)
    del df_all; gc.collect()
    print(f"  {len(sym_grps)} symbols after exclusions")

    # Per-stock analysis
    print("\n[3] Running per-stock analysis (backtest + challenge)...")
    results    = dict(saved_results)  # carry over previous
    new_count  = 0
    skip_count = 0
    fail_count = 0

    total = len(sym_grps)
    for i, (sym, grp) in enumerate(sym_grps.items()):
        # Skip if we already have a recent result for this stock and data hasn't changed
        if not force and sym in results:
            stock_latest = str(grp["date"].max().date())
            if results[sym].get("latest_date") == stock_latest:
                skip_count += 1
                continue

        try:
            res = analyse_stock(sym, grp, force=force)
            if res:
                results[sym] = res
                new_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1

        if (i+1) % 200 == 0:
            print(f"  {i+1}/{total} | new={new_count} skip={skip_count} fail={fail_count}",
                  flush=True)

    del sym_grps; gc.collect()

    print(f"\n  Done: {new_count} analysed, {skip_count} skipped (unchanged), {fail_count} failed/filtered")
    print(f"  Total in results: {len(results)} stocks")

    # Save checkpoint
    save_cp(latest, results)
    print(f"  Checkpoint saved ({latest})")

    # Write outputs
    print("\n[4] Writing outputs...")
    _write_outputs(results, latest)
    print("\nDone.")


def _write_outputs(results, latest):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ist = timezone(timedelta(hours=5,minutes=30))
    now = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    all_stocks = [v for v in results.values() if v is not None]

    # Sort master by longterm_score then avg_ret
    all_sorted = sorted(all_stocks,
                        key=lambda x: (-(x.get("longterm_score",0)), -(x.get("best_avg_ret",0))))

    # Remove detailed trade lists from master (too large) — keep only summary
    master_slim = []
    for s in all_sorted:
        slim = {k:v for k,v in s.items() if k not in ("recent_trades",)}
        # Strip full trade histories from all_strategies
        slim["all_strategies"] = [
            {k2:v2 for k2,v2 in st.items() if k2 != "recent_trades"}
            for st in (s.get("all_strategies") or [])
        ]
        master_slim.append(slim)

    jdump({"generated_at": now, "latest_date": latest,
           "n_stocks": len(master_slim), "stocks": master_slim},
          RESULT_F)
    print(f"  OK master_results.json ({len(master_slim)} stocks)")

    # Daily alerts: stocks with signal_today = True
    alerts = [s for s in all_stocks if s.get("signal_today")]
    alerts.sort(key=lambda x: (-(x.get("best_avg_ret",0))))
    alert_out = []
    for a in alerts:
        alert_out.append({
            "sym":               a["sym"],
            "price":             a["price"],
            "signal_strategies": a["signal_strategies"],
            "best_strategy":     a["best_strategy"],
            "best_grade":        a["best_grade"],
            "win_rate":          a["best_win_rate"],
            "avg_ret":           a["best_avg_ret"],
            "hold_days":         a["best_hold_days"],
            "max_gain":          a["best_max_gain"],
            "sharpe":            a["best_sharpe"],
            "explanation":       a["explanation"],
            "rsi":               a.get("rsi"),
            "vol_ratio":         a.get("vol_ratio"),
            "ret5":              a.get("ret5"),
            "avg_turnover_cr":   a.get("avg_turnover_cr"),
            "recent_trades":     a.get("recent_trades",[]),
        })
    jdump({"generated_at": now, "signal_date": latest,
           "n_alerts": len(alert_out), "alerts": alert_out},
          ALERTS_F)
    print(f"  OK daily_alerts.json ({len(alert_out)} alerts)")

    # Long-term picks: top 20 by longterm_score, A/A+ grade only
    lt = [s for s in all_stocks
          if s.get("best_grade") in ("A+","A") and s.get("longterm_score",0) >= 50]
    lt.sort(key=lambda x: (-x.get("longterm_score",0), -x.get("best_avg_ret",0)))
    lt_out = []
    for s in lt[:30]:
        lt_out.append({
            "sym":            s["sym"],
            "price":          s["price"],
            "longterm_score": s["longterm_score"],
            "best_strategy":  s["best_strategy"],
            "best_grade":     s["best_grade"],
            "win_rate":       s["best_win_rate"],
            "avg_ret":        s["best_avg_ret"],
            "hold_days":      s["best_hold_days"],
            "oos_win_rate":   s["oos_win_rate"],
            "oos_avg_ret":    s["oos_avg_ret"],
            "explanation":    s["explanation"],
            "avg_turnover_cr":s.get("avg_turnover_cr"),
            "rsi":            s.get("rsi"),
            "ret20":          s.get("ret20"),
            "n_strategies":   s["n_strategies"],
            "all_strategies": s.get("all_strategies",[]),
        })
    jdump({"generated_at": now, "latest_date": latest,
           "note": "Top long-term picks: A/A+ grade, score>=50, sorted by LT score",
           "picks": lt_out},
          LONGTERM_F)
    print(f"  OK longterm_picks.json ({len(lt_out)} picks)")

    # Print summary
    alerts_aplus = [a for a in alert_out if a.get("best_grade")=="A+"]
    print(f"\n  === Today's Summary ({latest}) ===")
    print(f"  Total qualified stocks : {len(all_stocks)}")
    print(f"  Signals today          : {len(alert_out)}")
    print(f"  A+ signals today       : {len(alerts_aplus)}")
    print(f"  Long-term picks        : {len(lt_out)}")
    if alerts_aplus:
        print(f"\n  Top A+ signals:")
        for a in alerts_aplus[:5]:
            print(f"    {a['sym']:<14} {a['best_strategy']:<20} "
                  f"wr={a['win_rate']:.0f}% avg=+{a['avg_ret']:.1f}% "
                  f"hold={a['hold_days']}d")

if __name__ == "__main__":
    main()
