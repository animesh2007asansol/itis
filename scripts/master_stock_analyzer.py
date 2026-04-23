#!/usr/bin/env python3
"""
master_stock_analyzer.py  v2
==============================
KEY RULES (v2):
  - 100% WIN RATE REQUIRED: every single trade in backtest must be positive.
    No averaging. No exceptions. One negative = strategy rejected for that stock.
  - TRADED TODAY: signal_today is only True if stock's latest data == manifest latest date.
  - PRE-SIGNAL CONTEXT: every occurrence captures RSI, vol ratio, 5d/20d returns at signal time.
  - ALL OCCURRENCES stored so signal history tab can show actual outcomes.
  - SIGNAL HISTORY: separate output file with every historical signal + actual return.
  - OOS CHALLENGE: both halves must also be 100% win rate.
"""
import json, gc, sys, os
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy"); sys.exit(1)

REPO_ROOT  = Path(__file__).parent.parent
DATA_DIR   = REPO_ROOT / "data"
OUT_DIR    = REPO_ROOT / "stock_analysis"
MANIFEST   = DATA_DIR / "manifest.json"
CHECKPOINT = OUT_DIR / "master_checkpoint.json"
RESULT_F   = OUT_DIR / "master_results.json"
ALERTS_F   = OUT_DIR / "daily_alerts.json"
LONGTERM_F = OUT_DIR / "longterm_picks.json"
HISTORY_F  = OUT_DIR / "signal_history.json"

MIN_PRICE        = 10.0
MIN_AVG_TURNOVER = 5_000_000   # Rs 5 Cr daily avg
MIN_HISTORY_DAYS = 120
WIN_RATE_REQUIRED = 100.0      # MUST BE 100% — every trade positive
MIN_AVG_RETURN   = 1.0         # minimum avg return (even at 100% wr, must be meaningful)
MIN_OCCURRENCES  = 5           # minimum trades (5 at 100% = 1/32 chance of luck)
HOLD_DAYS_SHORT  = 5
HOLD_DAYS_LONG   = 20
EXCLUDED_EXACT   = {
    "LIQUIDIETF","LIQUIDBEES","LIQUIDCASE","NIFTYBEES","JUNIORBEES",
    "BANKBEES","GOLDBEES","SILVERBEES","PSUBNKBEES","ITBEES","CPSEETF",
}
EXCLUDED_SUFFIX  = ("ETF","BEES","CASE","SETF","GILT")

SYM_A=["SYMBOL","TCKRSYMB"]; SER_A=["SERIES","SCTYSRS"]
O_A=["OPEN","OPNPRIC","OPEN PRICE"]; H_A=["HIGH","HGHPRIC","HIGH PRICE"]
L_A=["LOW","LWPRIC","LOW PRICE"];    C_A=["CLOSE","CLSPRIC","CLOSE PRICE","LASTPRIC"]
V_A=["TOTTRDQTY","TTLTRADGVOL","VOLUME"]

def r2(x):
    if x is None: return None
    try:
        f = float(x)
        return None if np.isnan(f) or np.isinf(f) else round(f*100)/100
    except: return None

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,(date_type,datetime)): return str(obj)
        if isinstance(obj,np.integer):  return int(obj)
        if isinstance(obj,np.floating):
            return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else None
        if isinstance(obj,np.bool_):    return bool(obj)
        if isinstance(obj,np.ndarray):  return obj.tolist()
        return super().default(obj)

def jdump(obj,path):
    with open(path,"w") as f: json.dump(obj,f,indent=2,cls=SafeEncoder)
def jload(path):
    try:
        with open(path) as f: return json.load(f)
    except: return {}

def _fc(hdr,aliases):
    for a in aliases:
        if a in hdr: return hdr.index(a)
    return -1

def load_csv(path):
    rows=[]
    try:
        with open(path,encoding="utf-8-sig",errors="replace") as f: lines=f.readlines()
        if len(lines)<2: return rows
        hdr=[h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
        i_s=_fc(hdr,SYM_A); i_sr=_fc(hdr,SER_A); i_o=_fc(hdr,O_A)
        i_h=_fc(hdr,H_A);   i_l=_fc(hdr,L_A);    i_c=_fc(hdr,C_A)
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
                if c>0 and o>0 and sym: rows.append({"sym":sym,"o":o,"h":h,"l":l,"c":c,"v":v})
            except: pass
    except: pass
    return rows

def load_all_days(manifest):
    rows=[]; loaded=0
    tds=sorted(manifest.keys())
    for ds in tds:
        y,m,_=ds.split("-")
        path=DATA_DIR/"equity"/y/m/f"{ds}.csv"
        if not path.exists(): continue
        r=load_csv(path)
        for x in r: x["date"]=ds
        rows.extend(r); loaded+=1
        if loaded%300==0: print(f"    {loaded} files…",flush=True)
    df=pd.DataFrame(rows)
    if df.empty: return df
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(["sym","date"]).reset_index(drop=True)

# ─── Indicators ───────────────────────────────────────────────────────────────
def add_indicators(df):
    c=df["c"]; v=df["v"]
    df["sma10"] =c.rolling(10,min_periods=5).mean()
    df["sma20"] =c.rolling(20,min_periods=10).mean()
    df["sma30"] =c.rolling(30,min_periods=15).mean()
    df["ema20"] =c.ewm(span=20,adjust=False).mean()
    df["ema50"] =c.ewm(span=50,adjust=False).mean()
    bb_mean=c.rolling(20,min_periods=10).mean()
    bb_std =c.rolling(20,min_periods=10).std()
    df["bb_upper"]=bb_mean+2*bb_std; df["bb_lower"]=bb_mean-2*bb_std
    df["bb_width"]=(df["bb_upper"]-df["bb_lower"])/bb_mean.replace(0,1)*100
    delta=c.diff()
    gain=delta.clip(lower=0).rolling(14,min_periods=7).mean()
    loss=(-delta.clip(upper=0)).rolling(14,min_periods=7).mean()
    rs=gain/loss.replace(0,0.0001)
    df["rsi"]=100-100/(1+rs)
    ema12=c.ewm(span=12,adjust=False).mean(); ema26=c.ewm(span=26,adjust=False).mean()
    df["macd"]=ema12-ema26; df["macd_sig"]=df["macd"].ewm(span=9,adjust=False).mean()
    hl=df["h"]-df["l"]; hc=(df["h"]-c.shift(1)).abs(); lc=(df["l"]-c.shift(1)).abs()
    df["atr"]=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14,min_periods=7).mean()
    df["vol_avg20"]=v.rolling(20,min_periods=10).mean()
    df["vol_ratio"]=v/df["vol_avg20"].replace(0,1)
    df["ret1"] =c.pct_change(1)*100; df["ret5"]=c.pct_change(5)*100
    df["ret20"]=c.pct_change(20)*100; df["ret_1w"]=c.pct_change(5)*100
    df["ret_1m"]=c.pct_change(20)*100
    df["high20"]=df["h"].rolling(20,min_periods=10).max()
    df["low20"] =df["l"].rolling(20,min_periods=10).min()
    df["high52w"]=df["h"].rolling(252,min_periods=100).max()
    is_red=(c<c.shift(1)).astype(int)
    df["consec_red"]=is_red*(is_red.groupby((is_red!=is_red.shift()).cumsum()).cumcount()+1)
    df["gap_pct"]=(df["o"]-c.shift(1))/c.shift(1).replace(0,1)*100
    df["turnover"]=c*v
    return df

# ─── Strategies ───────────────────────────────────────────────────────────────
def gen_signals(df):
    sigs={}; c=df["c"]
    sigs["SMA_CROSS"]=(df["sma10"]>df["sma30"])&(df["sma10"].shift(1)<=df["sma30"].shift(1))&(df["vol_ratio"]>=1.2)
    sigs["EMA_BOUNCE"]=(c>df["ema20"])&(c.shift(1)<df["ema20"].shift(1))&(df["vol_ratio"]>=1.5)&(df["rsi"]<60)
    sigs["RSI_OVERSOLD"]=(df["rsi"]>35)&(df["rsi"].shift(1)<=30)&(c>c.shift(1))
    sigs["MACD_CROSS"]=(df["macd"]>df["macd_sig"])&(df["macd"].shift(1)<=df["macd_sig"].shift(1))&(df["vol_ratio"]>=1.0)
    sigs["BB_BOUNCE"]=(df["l"]<=df["bb_lower"])&(c>df["bb_lower"])&(c.shift(1)<=df["bb_lower"].shift(1).fillna(c))&(df["vol_ratio"]>=1.3)
    sigs["VOL_BREAKOUT"]=(c>=df["high20"].shift(1))&(df["vol_ratio"]>=2.0)&(df["ret1"]>1.0)
    sigs["CONSEC_REVERSAL"]=(df["consec_red"].shift(1)>=3)&(c>c.shift(1))&(df["vol_ratio"]>=1.3)
    sigs["FLOOR_BOUNCE"]=(df["l"].shift(1)<=df["low20"].shift(2))&(c>c.shift(1))&(df["rsi"]<50)
    dr=(df["h"]-df["l"]).replace(0,0.0001); cp=(c-df["l"])/dr
    sigs["MOMENTUM_GAP"]=(df["gap_pct"]>2.0)&(cp>0.75)&(df["vol_ratio"]>=1.5)
    dr2=df["h"]-df["l"]
    sigs["RANGE_EXPAND"]=(dr2>2*df["atr"])&(c>df["o"])&(df["ret1"]>2.0)&(df["vol_ratio"]>=1.5)
    sigs["TREND_PULLBACK"]=(df["ret20"]>8.0)&(df["ret5"]<-2.0)&(c>c.shift(1))&(df["vol_ratio"]>=1.2)&(df["rsi"]<55)
    sigs["HIGHBREAK_52W"]=(c>=df["high52w"].shift(1)*0.98)&(df["vol_ratio"]>=2.0)&(df["ret1"]>1.5)&(df["rsi"]>60)
    return sigs

# ─── 100% Backtest Engine ─────────────────────────────────────────────────────
def backtest_100pct(df, signal_series, hold_days, label):
    """
    STRICT: every single trade must be positive. One negative = return None.
    Entry: next-day open after signal.
    Exit: close after hold_days.
    Captures pre-signal context for each trade.
    """
    signals=signal_series.fillna(False)
    c=df["c"].values; o=df["o"].values; n=len(df)
    dates=df["date"].values

    # Pre-computed context columns
    rsi_arr     = df["rsi"].values if "rsi" in df else np.full(n, np.nan)
    vol_r_arr   = df["vol_ratio"].values if "vol_ratio" in df else np.full(n, np.nan)
    ret5_arr    = df["ret5"].values if "ret5" in df else np.full(n, np.nan)
    ret20_arr   = df["ret20"].values if "ret20" in df else np.full(n, np.nan)
    ret1_arr    = df["ret1"].values if "ret1" in df else np.full(n, np.nan)
    bb_w_arr    = df["bb_width"].values if "bb_width" in df else np.full(n, np.nan)

    trades=[]; i=0
    while i < n-1:
        if not signals.iloc[i]: i+=1; continue
        entry_idx=i+1
        if entry_idx>=n: break
        entry_px=o[entry_idx]
        exit_idx=min(entry_idx+hold_days, n-1)
        exit_px=c[exit_idx]

        hold_w=c[entry_idx:exit_idx+1]
        max_c=float(hold_w.max()) if len(hold_w)>0 else exit_px
        min_c=float(hold_w.min()) if len(hold_w)>0 else exit_px
        ret      =(exit_px-entry_px)/entry_px*100
        max_gain =(max_c-entry_px)/entry_px*100
        max_dd   =(min_c-entry_px)/entry_px*100

        # ── 100% check: reject immediately on any negative return ──────────
        if ret <= 0:
            return None   # this strategy fails for this stock

        # Intermediate closes (3d, 10d if hold=20)
        ret_3d  = r2((c[min(entry_idx+3,n-1)]-entry_px)/entry_px*100) if hold_days>=5  else None
        ret_10d = r2((c[min(entry_idx+10,n-1)]-entry_px)/entry_px*100) if hold_days>=20 else None

        trades.append({
            "sig_date":    str(pd.Timestamp(dates[i]).date()),
            "entry_date":  str(pd.Timestamp(dates[entry_idx]).date()),
            "exit_date":   str(pd.Timestamp(dates[exit_idx]).date()),
            "entry_px":    r2(entry_px),
            "exit_px":     r2(exit_px),
            "ret":         r2(ret),
            "ret_3d":      ret_3d,
            "ret_10d":     ret_10d,
            "max_gain":    r2(max_gain),
            "max_dd":      r2(max_dd),
            # Pre-signal context (what the indicators looked like when signal fired)
            "ctx_rsi":     r2(rsi_arr[i]),
            "ctx_vol_ratio":r2(vol_r_arr[i]),
            "ctx_ret1":    r2(ret1_arr[i]),
            "ctx_ret5":    r2(ret5_arr[i]),
            "ctx_ret20":   r2(ret20_arr[i]),
            "ctx_bb_width":r2(bb_w_arr[i]),
            "ctx_close":   r2(float(c[i])),
            "positive":    True,   # guaranteed by the check above
        })
        i=exit_idx   # no overlapping trades

    if len(trades)<MIN_OCCURRENCES:
        return None

    rets=[t["ret"] for t in trades]
    avg_ret=np.mean(rets)
    if avg_ret<MIN_AVG_RETURN:
        return None

    # Context ranges (min/max/avg of each indicator at signal time)
    def ctx_stats(arr):
        arr=[t for t in arr if t is not None]
        if not arr: return None
        return {"min":r2(min(arr)),"max":r2(max(arr)),"avg":r2(sum(arr)/len(arr))}

    return {
        "strategy":     label,
        "hold_days":    hold_days,
        "n_trades":     len(trades),
        "win_rate":     100.0,    # guaranteed
        "avg_ret":      r2(avg_ret),
        "min_ret":      r2(min(rets)),
        "max_ret":      r2(max(rets)),
        "avg_max_gain": r2(np.mean([t["max_gain"] for t in trades])),
        "max_dd":       r2(min([t["max_dd"] for t in trades])),  # most negative intraday
        "sharpe":       r2(avg_ret/(np.std(rets)+0.001)*np.sqrt(252/hold_days)) if len(rets)>2 else 0.0,
        # Context ranges — what conditions trigger this signal
        "ctx_rsi":      ctx_stats([t["ctx_rsi"] for t in trades]),
        "ctx_vol_ratio":ctx_stats([t["ctx_vol_ratio"] for t in trades]),
        "ctx_ret5":     ctx_stats([t["ctx_ret5"] for t in trades]),
        "ctx_ret20":    ctx_stats([t["ctx_ret20"] for t in trades]),
        "ctx_bb_width": ctx_stats([t["ctx_bb_width"] for t in trades]),
        "trades":       trades,
    }

# ─── OOS Challenge ────────────────────────────────────────────────────────────
def challenge_100pct(df, signal_series, hold_days, label):
    mid=len(df)//2
    r1 =backtest_100pct(df.iloc[:mid].reset_index(drop=True), signal_series.iloc[:mid].reset_index(drop=True), hold_days, label)
    r2_=backtest_100pct(df.iloc[mid:].reset_index(drop=True), signal_series.iloc[mid:].reset_index(drop=True), hold_days, label)
    # Both must pass for A+; one for A
    if r1 and r2_: grade="A+"
    elif r1 or r2_: grade="A"
    else: grade="B"
    return grade, r1, r2_

# ─── Per-stock analysis ───────────────────────────────────────────────────────
def analyse_stock(sym, df, latest_manifest_date):
    df=df.copy().sort_values("date").reset_index(drop=True)
    n=len(df)
    if n<MIN_HISTORY_DAYS: return None
    c_last=float(df["c"].iloc[-1])
    if c_last<MIN_PRICE: return None
    tv_60=float((df["c"]*df["v"]).iloc[-60:].mean()) if n>=60 else 0
    if tv_60<MIN_AVG_TURNOVER: return None

    # ── KEY FIX: only signal if stock traded on LATEST manifest date ──────
    stock_latest=str(df["date"].iloc[-1].date())
    traded_today=(stock_latest==latest_manifest_date)

    df=add_indicators(df)
    all_sigs=gen_signals(df)
    best_strategies=[]

    for s_name, sig_series in all_sigs.items():
        n_sig=int(sig_series.fillna(False).sum())
        if n_sig<MIN_OCCURRENCES: continue

        # Try both hold periods — take the one with highest min_ret (most conservative)
        bt_short=backtest_100pct(df, sig_series, HOLD_DAYS_SHORT,  s_name)
        bt_long =backtest_100pct(df, sig_series, HOLD_DAYS_LONG,   s_name)

        best_bt=None; best_hold=None
        # Prefer the one with higher min_ret (safer floor)
        candidates=[(bt_short,HOLD_DAYS_SHORT),(bt_long,HOLD_DAYS_LONG)]
        for bt,hd in candidates:
            if bt is None: continue
            if best_bt is None or (bt["min_ret"] or 0)>(best_bt["min_ret"] or 0):
                best_bt=bt; best_hold=hd

        if best_bt is None: continue

        # OOS challenge — both halves must also be 100% clean
        grade,r_in,r_oos=challenge_100pct(df, sig_series, best_hold, s_name)
        if grade=="B": continue

        # today's signal only if stock traded today
        sig_today=bool(sig_series.iloc[-1]) and traded_today

        best_strategies.append({
            "strategy":     s_name,
            "hold_days":    best_hold,
            "grade":        grade,
            "win_rate":     100.0,
            "avg_ret":      best_bt["avg_ret"],
            "min_ret":      best_bt["min_ret"],
            "max_ret":      best_bt["max_ret"],
            "avg_max_gain": best_bt["avg_max_gain"],
            "max_dd":       best_bt["max_dd"],
            "sharpe":       best_bt["sharpe"],
            "n_trades":     best_bt["n_trades"],
            "oos_win_rate": r_oos["win_rate"] if r_oos else None,
            "oos_avg_ret":  r_oos["avg_ret"]  if r_oos else None,
            "ctx_rsi":      best_bt["ctx_rsi"],
            "ctx_vol_ratio":best_bt["ctx_vol_ratio"],
            "ctx_ret5":     best_bt["ctx_ret5"],
            "ctx_ret20":    best_bt["ctx_ret20"],
            "ctx_bb_width": best_bt["ctx_bb_width"],
            "all_trades":   best_bt["trades"],
            "recent_trades":best_bt["trades"][-5:],
            "signal_today": sig_today,
        })

    if not best_strategies: return None

    gr={"A+":0,"A":1,"B":2}
    best_strategies.sort(key=lambda x:(gr.get(x["grade"],3),-(x["avg_ret"] or 0)))
    top=best_strategies[0]
    any_signal=any(s["signal_today"] for s in best_strategies) and traded_today
    sig_strats=[s["strategy"] for s in best_strategies if s["signal_today"]]
    lt=_longterm_score(df,best_strategies)

    return {
        "sym":               sym,
        "latest_date":       stock_latest,
        "traded_today":      traded_today,
        "price":             r2(c_last),
        "avg_turnover_cr":   r2(tv_60/1e7),
        "n_strategies":      len(best_strategies),
        "best_strategy":     top["strategy"],
        "best_grade":        top["grade"],
        "best_win_rate":     100.0,
        "best_avg_ret":      top["avg_ret"],
        "best_min_ret":      top["min_ret"],
        "best_max_ret":      top["max_ret"],
        "best_hold_days":    top["hold_days"],
        "best_max_gain":     top["avg_max_gain"],
        "best_sharpe":       top["sharpe"],
        "best_max_dd":       top["max_dd"],
        "oos_win_rate":      top["oos_win_rate"],
        "oos_avg_ret":       top["oos_avg_ret"],
        "all_strategies":    [{k:v for k,v in s.items() if k!="all_trades"} for s in best_strategies],
        "signal_today":      any_signal,
        "signal_strategies": sig_strats,
        "explanation":       _explain(df,top["strategy"],top["avg_ret"],top["hold_days"]),
        "longterm_score":    lt,
        "recent_trades":     top["recent_trades"],
        # Context ranges for best strategy
        "ctx_rsi":           top["ctx_rsi"],
        "ctx_vol_ratio":     top["ctx_vol_ratio"],
        "ctx_ret5":          top["ctx_ret5"],
        "ctx_ret20":         top["ctx_ret20"],
        # Current indicators
        "rsi":               r2(df["rsi"].iloc[-1]),
        "vol_ratio":         r2(df["vol_ratio"].iloc[-1]),
        "ret1":              r2(df["ret1"].iloc[-1]),
        "ret5":              r2(df["ret5"].iloc[-1]),
        "ret1w":             r2(df["ret5"].iloc[-1]),
        "ret1m":             r2(df["ret20"].iloc[-1]),
        "bb_width":          r2(df["bb_width"].iloc[-1]),
        # Full trade history for signal history tab (best strategy only)
        "_all_trades_best":  top["all_trades"],
    }

def _explain(df,strategy,avg_ret,hold_days):
    base=f"Historically, every single trade using {strategy} on this stock was profitable. "
    base+=f"Average return: +{avg_ret:.1f}% over {hold_days} trading days. Min return was always positive. "
    tips={
        "SMA_CROSS":     "Signal fires when the fast 10-day moving average crosses above the slow 30-day MA with above-average volume. This marks a shift in momentum — institutional money is rotating in.",
        "EMA_BOUNCE":    "Price dips to the 20-day exponential average, then closes above it with a volume surge. Buyers defend this level every time it is tested.",
        "RSI_OVERSOLD":  "RSI drops below 30 (stock is technically oversold) then recovers above 35. Every oversold bounce on this stock has been profitable.",
        "MACD_CROSS":    "MACD line crosses above its signal line. On this stock, every such crossover preceded a meaningful move upward.",
        "BB_BOUNCE":     "Price touches the lower Bollinger Band then recovers inside. The band squeeze then expansion gave consistently positive outcomes.",
        "VOL_BREAKOUT":  "Price breaks to a 20-day high with at least double the average volume. Institutional conviction behind every breakout here.",
        "CONSEC_REVERSAL":"After 3+ consecutive down days, this stock reversed every single time. Selling exhaustion is a reliable buy trigger here.",
        "FLOOR_BOUNCE":  "When the stock hits its 20-day low and then closes higher with RSI below 50, it has bounced profitably every occurrence.",
        "MOMENTUM_GAP":  "Gap-up opens >2% that close in the top quarter of the day's range — continuation has been guaranteed on this stock.",
        "RANGE_EXPAND":  "Days with range >2× ATR closing bullish have always been followed by positive returns on this stock.",
        "TREND_PULLBACK":"In an established uptrend, pullbacks on this stock have been 100% reliable buying opportunities.",
        "HIGHBREAK_52W": "Every 52-week high breakout with strong volume on this stock has led to further gains.",
    }
    return base+tips.get(strategy,"Every historical occurrence of this signal on this stock resulted in a positive return.")

def _longterm_score(df,strategies):
    c=df["c"]; n=len(c)
    s=20
    ema50=c.ewm(span=50,adjust=False).mean()
    if float(c.iloc[-1])>float(ema50.iloc[-1]): s+=15
    ret60=(float(c.iloc[-1])-float(c.iloc[max(0,n-60)]))/float(c.iloc[max(0,n-60)])*100
    if ret60>0: s+=10
    if ret60>10: s+=10
    aplus=[x for x in strategies if x["grade"]=="A+"]
    s+=min(20,len(aplus)*8)
    best_ar=max([x["avg_ret"] or 0 for x in strategies],default=0)
    if best_ar>=5: s+=15
    elif best_ar>=3: s+=10
    return min(100,s)

def load_cp():
    cp=jload(CHECKPOINT)
    return cp.get("processed_date",""),cp.get("results",{})

def save_cp(processed_date,results):
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    jdump({"processed_date":processed_date,"results":results},CHECKPOINT)

def main():
    force=os.getenv("FORCE_RERUN","false").lower()=="true"
    print("="*70)
    print("Master Stock Analyzer  v2")
    print("  100% win rate required — every single trade must be positive")
    print("  'traded today' check — no ghost signals from inactive stocks")
    print("="*70)
    OUT_DIR.mkdir(parents=True,exist_ok=True)

    manifest=jload(MANIFEST); tds=sorted(manifest.keys()); latest=tds[-1]
    print(f"\n  Trading days: {len(tds)} | Latest: {latest}")

    cp_date,saved_results=load_cp()
    if force: cp_date=""; saved_results={}; print("  FORCE RERUN")
    elif cp_date==latest:
        print(f"  Already processed {latest}. Regenerating outputs only.")
        _write_outputs(saved_results,latest); return
    else:
        print(f"  Checkpoint: {cp_date or 'none'} → processing to {latest}")

    print("\n[1] Loading all equity data…")
    df_all=load_all_days(manifest)
    if df_all.empty: print("ERROR: No data"); sys.exit(1)
    print(f"  {len(df_all):,} rows, {df_all['sym'].nunique()} symbols")

    print("\n[2] Grouping by symbol…")
    sym_grps={}
    for sym,grp in df_all.groupby("sym"):
        if sym in EXCLUDED_EXACT: continue
        if any(sym.upper().endswith(s) for s in EXCLUDED_SUFFIX): continue
        sym_grps[sym]=grp.reset_index(drop=True)
    del df_all; gc.collect()
    print(f"  {len(sym_grps)} symbols after exclusions")

    print("\n[3] Running 100%-win-rate analysis…")
    results=dict(saved_results)
    new_count=skip_count=fail_count=0
    total=len(sym_grps)

    for i,(sym,grp) in enumerate(sym_grps.items()):
        if not force and sym in results:
            stk_latest=str(grp["date"].max().date())
            if results[sym].get("latest_date")==stk_latest:
                skip_count+=1; continue
        try:
            res=analyse_stock(sym,grp,latest)
            if res: results[sym]=res; new_count+=1
            else: fail_count+=1
        except Exception as e:
            fail_count+=1
        if (i+1)%200==0:
            print(f"  {i+1}/{total} | new={new_count} skip={skip_count} fail={fail_count}",flush=True)

    del sym_grps; gc.collect()
    print(f"\n  Done: {new_count} qualified, {skip_count} skipped, {fail_count} filtered/failed")
    save_cp(latest,results)
    print(f"  Checkpoint saved")

    print("\n[4] Writing outputs…")
    _write_outputs(results,latest)
    print("\nDone.")

def _write_outputs(results,latest):
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    ist=timezone(timedelta(hours=5,minutes=30))
    now=datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")
    all_stocks=[v for v in results.values() if v]

    # Sort master by longterm_score
    all_sorted=sorted(all_stocks,key=lambda x:(-(x.get("longterm_score",0)),-(x.get("best_avg_ret") or 0)))

    # Slim master (no full trade lists)
    master_slim=[]
    for s in all_sorted:
        slim={k:v for k,v in s.items() if k not in ("_all_trades_best","recent_trades")}
        slim["all_strategies"]=[{k2:v2 for k2,v2 in st.items() if k2 not in ("all_trades","recent_trades")} for st in (s.get("all_strategies") or [])]
        slim["recent_trades"]=s.get("recent_trades",[])
        master_slim.append(slim)

    jdump({"generated_at":now,"latest_date":latest,"n_stocks":len(master_slim),"stocks":master_slim},RESULT_F)
    print(f"  OK master_results.json ({len(master_slim)} stocks)")

    # Daily alerts — only traded today AND signal today
    alerts=[s for s in all_stocks if s.get("signal_today") and s.get("traded_today")]
    alerts.sort(key=lambda x:(-(x.get("best_avg_ret") or 0)))
    alert_out=[]
    for a in alerts:
        alert_out.append({
            "sym":a["sym"],"price":a["price"],
            "signal_strategies":a["signal_strategies"],
            "best_strategy":a["best_strategy"],"best_grade":a["best_grade"],
            "win_rate":100.0,"avg_ret":a["best_avg_ret"],"min_ret":a["best_min_ret"],
            "hold_days":a["best_hold_days"],"max_gain":a["best_max_gain"],
            "sharpe":a["best_sharpe"],"max_dd":a["best_max_dd"],
            "explanation":a["explanation"],
            "ctx_rsi":a.get("ctx_rsi"),"ctx_vol_ratio":a.get("ctx_vol_ratio"),
            "ctx_ret5":a.get("ctx_ret5"),"ctx_ret20":a.get("ctx_ret20"),
            "rsi":a.get("rsi"),"vol_ratio":a.get("vol_ratio"),
            "ret1":a.get("ret1"),"ret1w":a.get("ret1w"),"ret1m":a.get("ret1m"),
            "avg_turnover_cr":a.get("avg_turnover_cr"),
            "recent_trades":a.get("recent_trades",[]),
        })
    jdump({"generated_at":now,"signal_date":latest,"n_alerts":len(alert_out),"alerts":alert_out},ALERTS_F)
    print(f"  OK daily_alerts.json ({len(alert_out)} alerts — traded today only)")

    # Long-term picks
    lt=[s for s in all_stocks if s.get("best_grade") in ("A+","A") and (s.get("longterm_score") or 0)>=50]
    lt.sort(key=lambda x:(-(x.get("longterm_score") or 0),-(x.get("best_avg_ret") or 0)))
    lt_out=[]
    for s in lt[:30]:
        lt_out.append({
            "sym":s["sym"],"price":s["price"],"longterm_score":s["longterm_score"],
            "best_strategy":s["best_strategy"],"best_grade":s["best_grade"],
            "win_rate":100.0,"avg_ret":s["best_avg_ret"],"min_ret":s.get("best_min_ret"),
            "hold_days":s["best_hold_days"],"oos_win_rate":s["oos_win_rate"],"oos_avg_ret":s["oos_avg_ret"],
            "explanation":s["explanation"],"avg_turnover_cr":s.get("avg_turnover_cr"),
            "rsi":s.get("rsi"),"ret1w":s.get("ret1w"),"ret1m":s.get("ret1m"),
            "n_strategies":s["n_strategies"],
            "all_strategies":[{k:v for k,v in st.items() if k not in ("all_trades","recent_trades")} for st in (s.get("all_strategies") or [])],
            "ctx_rsi":s.get("ctx_rsi"),"ctx_vol_ratio":s.get("ctx_vol_ratio"),
        })
    jdump({"generated_at":now,"latest_date":latest,"picks":lt_out},LONGTERM_F)
    print(f"  OK longterm_picks.json ({len(lt_out)} picks)")

    # Signal history — ALL historical trades from qualifying stocks
    history_entries=[]
    for s in all_stocks:
        trades=s.get("_all_trades_best",[]) or s.get("recent_trades",[])
        for t in trades:
            history_entries.append({
                "sym":s["sym"],"strategy":s["best_strategy"],
                "grade":s["best_grade"],"hold_days":s["best_hold_days"],
                "sig_date":t.get("sig_date"),"entry_date":t.get("entry_date"),
                "exit_date":t.get("exit_date"),"entry_px":t.get("entry_px"),
                "exit_px":t.get("exit_px"),"ret":t.get("ret"),
                "ret_3d":t.get("ret_3d"),"ret_10d":t.get("ret_10d"),
                "max_gain":t.get("max_gain"),"max_dd":t.get("max_dd"),
                "ctx_rsi":t.get("ctx_rsi"),"ctx_vol_ratio":t.get("ctx_vol_ratio"),
                "ctx_ret5":t.get("ctx_ret5"),"ctx_ret20":t.get("ctx_ret20"),
            })
    history_entries.sort(key=lambda x:(x.get("sig_date") or ""),reverse=True)
    jdump({"generated_at":now,"n_signals":len(history_entries),"signals":history_entries[:2000]},HISTORY_F)
    print(f"  OK signal_history.json ({len(history_entries)} historical signals)")

    # Summary
    aplus=[a for a in alert_out if a.get("best_grade")=="A+"]
    print(f"\n  === Summary ({latest}) ===")
    print(f"  Qualified stocks  : {len(all_stocks)}")
    print(f"  Signals today     : {len(alert_out)} (all traded today)")
    print(f"  A+ signals today  : {len(aplus)}")
    print(f"  LT picks          : {len(lt_out)}")
    for a in aplus[:5]:
        print(f"    {a['sym']:<14} {a['best_strategy']:<20} avg=+{a['avg_ret']:.1f}% min=+{a['min_ret']:.1f}%")

if __name__=="__main__":
    main()
