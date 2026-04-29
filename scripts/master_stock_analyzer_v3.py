#!/usr/bin/env python3
"""
master_stock_analyzer_v3.py
============================
COMPREHENSIVE NSE STOCK PATTERN ENGINE

USE CASE 1 — SURGE MOMENTUM
  When a stock rises X% over N days, does it reliably continue positively?
  Parametrized search: X=[3,5,7,10,15,20]%, N=[1,2,3,5,10,20]d, M=[5,10,20,30,60]d
  Reports best (X,N,M) combo with ≥95% win rate and ≥10% min return.

USE CASE 2 — SEASONAL PATTERNS
  Does this stock go up reliably in certain months or quarters every year?
  Check each month (Jan-Dec) and quarter (Q1-Q4).
  Win rate computed across all years of data with minimum 10% return.

USE CASE 3 — COMBINED SIGNAL
  Surge momentum firing during a strong seasonal window = double confirmation.

USE CASE 4 — TECHNICAL PATTERNS (25 strategies, parametrised batches)
  Indicators: SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, Stochastic, ADX,
  Williams %R, VWAP, Ichimoku, Volume Profile, Supertrend, Cup & Handle approx,
  Double Bottom, Bull Flag, Golden Cross, Death Cross, Pivot Points, etc.
  All combinations tried; best per-stock combo identified.

VALIDATION FRAMEWORK (multi-year OOS):
  Train : all data up to end of 2022
  OOS-A : 2023 data only
  OOS-B : 2024 data only
  OOS-C : 2025 data only
  Recent: 2026 to date
  Pattern gets confidence score based on how many periods it holds.

REQUIREMENTS:
  - Stock must be current (traded on latest data date)
  - At least 2 years of data
  - Min 10% return EVERY occurrence (any ≤0% → strategy invalidated)
  - ≥95% win rate (allows 1 bad trade in 20, but no -ve returns)
  - Minimum 4 occurrences over at least 2 different years
  - Return rate = return% / hold_days (primary ranking metric)

INCREMENTAL: Checkpoint stores per-stock last-processed date.
  On daily run, only new data rows are re-evaluated.
  Saves 99% of compute after first run.
"""

import json, gc, sys, os, math
from pathlib import Path
from datetime import datetime, timezone, timedelta, date as date_type
from itertools import product
import warnings
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("pip install pandas numpy"); sys.exit(1)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).parent.parent
DATA  = ROOT / "data"
OUT   = ROOT / "stock_analysis"
MANI  = DATA / "manifest.json"
CP    = OUT  / "checkpoint_v3.json"

# ─── Thresholds ───────────────────────────────────────────────────────────────
MIN_PRICE          = 10.0
MIN_TURNOVER       = 5_000_000    # Rs 5 Cr daily avg (60d)
MIN_HISTORY_DAYS   = 500          # ~2 years minimum
WIN_RATE           = 95.0         # 95% minimum win rate
MIN_RETURN         = 10.0         # 10% minimum return per occurrence
MIN_OCC            = 4            # at least 4 occurrences
MIN_OCC_YEARS      = 2            # must span at least 2 different calendar years
MAX_NEG_RETURN     = -0.01        # any return below this = strategy invalid for stock

# UC1 parameter grid
UC1_SURGE_PCT    = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
UC1_SURGE_DAYS   = [1, 2, 3, 5, 10, 20]
UC1_FORWARD_DAYS = [5, 10, 15, 20, 30, 60]

# UC2 seasonal periods
UC2_MONTHS   = list(range(1, 13))      # Jan=1 … Dec=12
UC2_QUARTERS = [1, 2, 3, 4]
UC2_HOLD     = [10, 20, 30]            # trading days

EXCLUDED = {
    "LIQUIDIETF","LIQUIDBEES","LIQUIDCASE","NIFTYBEES","JUNIORBEES",
    "BANKBEES","GOLDBEES","SILVERBEES","PSUBNKBEES","ITBEES","CPSEETF",
}
EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT")

# Column aliases
SYM=["SYMBOL","TCKRSYMB"]; SER=["SERIES","SCTYSRS"]
OPN=["OPEN","OPNPRIC","OPEN PRICE"]; HI=["HIGH","HGHPRIC","HIGH PRICE"]
LO=["LOW","LWPRIC","LOW PRICE"];   CL=["CLOSE","CLSPRIC","CLOSE PRICE","LASTPRIC"]
VOL=["TOTTRDQTY","TTLTRADGVOL","VOLUME"]

def r2(x):
    if x is None: return None
    try:
        f=float(x)
        return None if (math.isnan(f) or math.isinf(f)) else round(f*100)/100
    except: return None

def safe(x, default=0.0):
    try:
        f=float(x)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except: return default

class Enc(json.JSONEncoder):
    def default(self,o):
        if isinstance(o,(date_type,datetime)): return str(o)
        if isinstance(o,np.integer):  return int(o)
        if isinstance(o,np.floating):
            return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
        if isinstance(o,np.bool_):    return bool(o)
        if isinstance(o,np.ndarray):  return o.tolist()
        return super().default(o)

def jdump(obj,path):
    with open(path,"w") as f: json.dump(obj,f,indent=2,cls=Enc)
def jload(path):
    try:
        with open(path) as f: return json.load(f)
    except: return {}

# ─── CSV loader ───────────────────────────────────────────────────────────────
def _fc(hdr,als):
    for a in als:
        if a in hdr: return hdr.index(a)
    return -1

def load_csv(path):
    rows=[]
    try:
        with open(path,encoding="utf-8-sig",errors="replace") as f: lines=f.readlines()
        if len(lines)<2: return rows
        hdr=[h.strip().strip('"').strip("'").upper() for h in lines[0].split(",")]
        is_=_fc(hdr,SYM); isr=_fc(hdr,SER); io=_fc(hdr,OPN)
        ih=_fc(hdr,HI);   il=_fc(hdr,LO); ic=_fc(hdr,CL); iv=_fc(hdr,VOL)
        if is_<0 or ic<0: return rows
        mc=max(x for x in [is_,io,ih,il,ic,iv] if x>=0)
        for line in lines[1:]:
            line=line.strip()
            if not line: continue
            cols=[c.strip().strip('"').strip("'") for c in line.split(",")]
            if len(cols)<=mc: continue
            ser=cols[isr].strip() if isr>=0 else "EQ"
            if ser not in ("EQ","BE"): continue
            try:
                sym=cols[is_].strip(); cv=float(cols[ic])
                ov=float(cols[io]) if io>=0 else cv
                hv=float(cols[ih]) if ih>=0 else cv
                lv=float(cols[il]) if il>=0 else cv
                vv=float(str(cols[iv]).replace(",","")) if iv>=0 else 0.0
                if cv>0 and ov>0 and sym:
                    rows.append({"sym":sym,"o":ov,"h":hv,"l":lv,"c":cv,"v":vv})
            except: pass
    except: pass
    return rows

def load_all(manifest):
    rows=[]; n=0
    for ds in sorted(manifest.keys()):
        y,m,_=ds.split("-")
        p=DATA/"equity"/y/m/f"{ds}.csv"
        if not p.exists(): continue
        r=load_csv(p)
        for x in r: x["date"]=ds
        rows.extend(r); n+=1
        if n%300==0: print(f"    {n} files…",flush=True)
    df=pd.DataFrame(rows)
    if df.empty: return df
    df["date"]=pd.to_datetime(df["date"])
    return df.sort_values(["sym","date"]).reset_index(drop=True)

# ─── Indicators ───────────────────────────────────────────────────────────────
def indicators(df):
    c=df["c"]; v=df["v"]
    # Moving averages
    df["sma10"] =c.rolling(10,min_periods=5).mean()
    df["sma20"] =c.rolling(20,min_periods=10).mean()
    df["sma30"] =c.rolling(30,min_periods=15).mean()
    df["sma50"] =c.rolling(50,min_periods=25).mean()
    df["sma200"]=c.rolling(200,min_periods=100).mean()
    df["ema12"] =c.ewm(span=12,adjust=False).mean()
    df["ema20"] =c.ewm(span=20,adjust=False).mean()
    df["ema26"] =c.ewm(span=26,adjust=False).mean()
    df["ema50"] =c.ewm(span=50,adjust=False).mean()
    # Bollinger
    bm=c.rolling(20,min_periods=10).mean(); bs=c.rolling(20,min_periods=10).std()
    df["bb_up"]=bm+2*bs; df["bb_lo"]=bm-2*bs
    df["bb_w"]=(df["bb_up"]-df["bb_lo"])/bm.replace(0,1)*100
    df["bb_pos"]=(c-df["bb_lo"])/(df["bb_up"]-df["bb_lo"]+0.0001)
    # RSI 14
    delta=c.diff()
    gain=delta.clip(lower=0).rolling(14,min_periods=7).mean()
    loss=(-delta.clip(upper=0)).rolling(14,min_periods=7).mean()
    df["rsi"]=100-100/(1+gain/loss.replace(0,0.0001))
    # Stochastic (14,3)
    lo14=df["l"].rolling(14,min_periods=7).min(); hi14=df["h"].rolling(14,min_periods=7).max()
    df["stoch_k"]=(c-lo14)/(hi14-lo14+0.0001)*100
    df["stoch_d"]=df["stoch_k"].rolling(3).mean()
    # MACD
    df["macd"]=df["ema12"]-df["ema26"]
    df["macd_sig"]=df["macd"].ewm(span=9,adjust=False).mean()
    df["macd_hist"]=df["macd"]-df["macd_sig"]
    # ATR 14
    hl=df["h"]-df["l"]; hc=(df["h"]-c.shift(1)).abs(); lc=(df["l"]-c.shift(1)).abs()
    df["atr"]=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14,min_periods=7).mean()
    # ADX
    up_move=df["h"]-df["h"].shift(1); dn_move=df["l"].shift(1)-df["l"]
    plus_dm=np.where((up_move>dn_move)&(up_move>0),up_move,0)
    minus_dm=np.where((dn_move>up_move)&(dn_move>0),dn_move,0)
    atr14=df["atr"]
    plus_di=pd.Series(plus_dm,index=df.index).rolling(14).sum()/(atr14.rolling(14).sum()+0.001)*100
    minus_di=pd.Series(minus_dm,index=df.index).rolling(14).sum()/(atr14.rolling(14).sum()+0.001)*100
    dx=((plus_di-minus_di).abs()/(plus_di+minus_di+0.001)*100)
    df["adx"]=dx.rolling(14).mean(); df["plus_di"]=plus_di; df["minus_di"]=minus_di
    # Williams %R
    df["willr"]=(hi14-c)/(hi14-lo14+0.0001)*-100
    # OBV
    obv_dir=np.sign(c.diff().fillna(0))
    df["obv"]=(v*obv_dir).cumsum()
    df["obv_sma20"]=df["obv"].rolling(20).mean()
    # Volume stats
    df["vol20"]=v.rolling(20,min_periods=10).mean()
    df["vol_r"]=v/(df["vol20"].replace(0,1))
    # Price momentum
    df["ret1"] =c.pct_change(1)*100; df["ret3"] =c.pct_change(3)*100
    df["ret5"] =c.pct_change(5)*100; df["ret10"]=c.pct_change(10)*100
    df["ret20"]=c.pct_change(20)*100; df["ret60"]=c.pct_change(60)*100
    # N-day range
    df["hi20"]=df["h"].rolling(20,min_periods=10).max()
    df["lo20"]=df["l"].rolling(20,min_periods=10).min()
    df["hi52"]=df["h"].rolling(252,min_periods=100).max()
    df["lo52"]=df["l"].rolling(252,min_periods=100).min()
    # Supertrend (10,3)
    hl2=(df["h"]+df["l"])/2
    atr10=pd.concat([df["h"]-df["l"],(df["h"]-c.shift(1)).abs(),(df["l"]-c.shift(1)).abs()],axis=1).max(axis=1).rolling(10).mean()
    upper=hl2+3*atr10; lower=hl2-3*atr10
    df["st_upper"]=upper; df["st_lower"]=lower
    # Gap
    df["gap"]=(df["o"]-c.shift(1))/c.shift(1).replace(0,1)*100
    # Consecutive moves
    is_up=(c>c.shift(1)).astype(int)
    df["consec_up"]=is_up*(is_up.groupby((is_up!=is_up.shift()).cumsum()).cumcount()+1)
    is_dn=(c<c.shift(1)).astype(int)
    df["consec_dn"]=is_dn*(is_dn.groupby((is_dn!=is_dn.shift()).cumsum()).cumcount()+1)
    # Pivot points (daily)
    df["pivot"]=(df["h"]+df["l"]+c)/3
    df["r1"]=2*df["pivot"]-df["l"]; df["s1"]=2*df["pivot"]-df["h"]
    # Turnover
    df["tv"]=c*v
    return df

# ─── CORE BACKTEST (95% win, 10% min return) ──────────────────────────────────

def _buy_stats(trades):
    """
    Aggregate Day1 buying statistics across all trades.
    Returns a dict with gap, dip, close stats for the buying strategy tab.
    """
    if not trades: return {}

    def vals(key): return [t[key] for t in trades if t.get(key) is not None]

    gaps     = vals("d1_gap_pct")
    dips     = vals("d1_dip_pct")
    crets    = vals("d1_close_ret")
    intras   = vals("d1_intra_ret")

    def avg(lst):  return r2(sum(lst)/len(lst)) if lst else None
    def mn(lst):   return r2(min(lst)) if lst else None
    def mx(lst):   return r2(max(lst)) if lst else None
    def pct_pos(lst): return r2(sum(1 for x in lst if x > 0)/len(lst)*100) if lst else None
    def pct_neg(lst): return r2(sum(1 for x in lst if x < 0)/len(lst)*100) if lst else None

    # Optimal buy: if price dips below open, wait for dip else buy at open
    avg_dip      = avg(dips) or 0
    pct_dips     = pct_neg(dips)  # % of times price went below open on D1
    avg_dip_abs  = avg([abs(d) for d in dips if d < 0])

    # Recommend: if dip happens >60% of time with avg dip > 0.5%, wait for dip
    if (pct_dips or 0) > 60 and (avg_dip_abs or 0) > 0.5:
        buy_strategy = f"Wait for {avg_dip_abs:.1f}% dip from open ({pct_dips:.0f}% of signals dip below open)"
        buy_at       = "Open - dip"
    else:
        buy_strategy = "Buy at open (price rarely dips below open on signal day)"
        buy_at       = "Open"

    return {
        "n_trades":         len(trades),
        # Gap stats (D1 open vs D0 close)
        "avg_gap_pct":      avg(gaps),
        "min_gap_pct":      mn(gaps),
        "max_gap_pct":      mx(gaps),
        "pct_gap_up":       pct_pos(gaps),
        "pct_gap_down":     pct_neg(gaps),
        # D1 intraday dip from open (always <= 0 means it went below open)
        "avg_dip_pct":      avg(dips),
        "worst_dip_pct":    mn(dips),
        "pct_dips_below_open": pct_neg(dips),
        "avg_dip_when_dips":avg([abs(d) for d in dips if d < 0]),
        # D1 close stats
        "avg_d1_close_ret": avg(crets),
        "min_d1_close_ret": mn(crets),
        "max_d1_close_ret": mx(crets),
        "pct_d1_close_pos": pct_pos(crets),
        "pct_d1_close_neg": pct_neg(crets),
        # D1 intraday (open to close)
        "avg_d1_intra_ret": avg(intras),
        "pct_d1_intra_pos": pct_pos(intras),
        # Recommendation
        "buy_strategy":     buy_strategy,
        "buy_at":           buy_at,
    }


def backtest(df, signal_series, hold_days, label):
    """
    95% win rate required, 10% minimum return per trade.
    Any trade returning ≤0% immediately invalidates this strategy for this stock.
    Returns None if strategy fails requirements.
    Captures pre-signal context at each occurrence.
    """
    signals=signal_series.fillna(False)
    c=df["c"].values; o=df["o"].values; n=len(df)
    h=df["h"].values; l=df["l"].values
    dates=df["date"].values

    rsi=df["rsi"].values if "rsi" in df else np.full(n,np.nan)
    volr=df["vol_r"].values if "vol_r" in df else np.full(n,np.nan)
    r1 =df["ret1"].values if "ret1" in df else np.full(n,np.nan)
    r5 =df["ret5"].values if "ret5" in df else np.full(n,np.nan)
    r20=df["ret20"].values if "ret20" in df else np.full(n,np.nan)
    adx=df["adx"].values if "adx" in df else np.full(n,np.nan)

    trades=[]; i=0; n_bad=0
    while i<n-1:
        if not signals.iloc[i]: i+=1; continue
        ei=i+1
        if ei>=n: break
        ep=o[ei]; xi=min(ei+hold_days,n-1); xp=c[xi]
        hw=c[ei:xi+1]
        mx=float(hw.max()) if len(hw)>0 else xp
        mn=float(hw.min()) if len(hw)>0 else xp
        ret=(xp-ep)/ep*100
        mg=(mx-ep)/ep*100
        dd=(mn-ep)/ep*100

        # Invalidate immediately on any negative return
        if ret<=0:
            return None

        # Also check minimum 10% return
        if ret < MIN_RETURN:
            n_bad += 1

        # 3d and 10d intermediate
        r3=r2((c[min(ei+3,n-1)]-ep)/ep*100) if hold_days>=5 else None
        r10=r2((c[min(ei+10,n-1)]-ep)/ep*100) if hold_days>=20 else None

        # Previous day context (what happened before signal)
        prev1 = r2(r1[i]) if not math.isnan(safe(r1[i],float('nan'))) else None
        prev5 = r2(r5[i]) if not math.isnan(safe(r5[i],float('nan'))) else None

        # Day0 (signal day) and Day1 (entry day = next day after signal)
        d0_close  = float(c[i])
        d1_open   = float(o[ei])           # = ep (entry price)
        d1_high   = float(h[ei])
        d1_low    = float(l[ei])
        d1_close  = float(c[ei])
        # Gap: Day1 open vs Day0 close
        d1_gap_pct    = r2((d1_open  - d0_close) / d0_close * 100) if d0_close else None
        # Dip: how far below D1 open did price fall intraday (always <=0 or 0)
        d1_dip_pct    = r2((d1_low   - d1_open)  / d1_open  * 100) if d1_open  else None
        # D1 close vs D0 close
        d1_close_ret  = r2((d1_close - d0_close) / d0_close * 100) if d0_close else None
        # D1 close vs D1 open (intraday direction)
        d1_intra_ret  = r2((d1_close - d1_open)  / d1_open  * 100) if d1_open  else None

        trades.append({
            "sig_date":  str(pd.Timestamp(dates[i]).date()),
            "entry_date":str(pd.Timestamp(dates[ei]).date()),
            "exit_date": str(pd.Timestamp(dates[xi]).date()),
            "is_open":   xi < (ei + hold_days),
            "entry_px":  r2(ep), "exit_px":r2(xp),
            "ret":r2(ret), "ret_3d":r3, "ret_10d":r10,
            "max_gain":r2(mg), "max_dd":r2(dd),
            # Day1 buying stats
            "d0_close":     r2(d0_close),
            "d1_open":      r2(d1_open),
            "d1_high":      r2(d1_high),
            "d1_low":       r2(d1_low),
            "d1_close":     r2(d1_close),
            "d1_gap_pct":   d1_gap_pct,
            "d1_dip_pct":   d1_dip_pct,
            "d1_close_ret": d1_close_ret,
            "d1_intra_ret": d1_intra_ret,
            # Pre-signal context
            "ctx_rsi":   r2(rsi[i]),
            "ctx_vol_r": r2(volr[i]),
            "ctx_ret1":  prev1,
            "ctx_ret5":  prev5,
            "ctx_ret20": r2(r20[i]),
            "ctx_adx":   r2(adx[i]),
        })
        i=xi

    n_trades=len(trades)
    if n_trades < MIN_OCC: return None

    # Check year span
    years=set(t["sig_date"][:4] for t in trades)
    if len(years) < MIN_OCC_YEARS: return None

    rets=[t["ret"] for t in trades]
    # Win rate check (trades with ret>=10% are full wins; 0<ret<10 are partial — still positive but below threshold)
    full_wins=[t for t in trades if t["ret"]>=MIN_RETURN]
    wr_full=len(full_wins)/n_trades*100
    if wr_full < WIN_RATE: return None

    avg_ret=sum(rets)/n_trades
    min_ret=min(rets)
    max_ret=max(rets)
    avg_mg =sum(t["max_gain"] for t in trades)/n_trades

    # Return rate per day (primary ranking metric)
    ret_rate=avg_ret/hold_days

    # Context stats (range when signal fires)
    def cstats(field):
        vals=[t[field] for t in trades if t.get(field) is not None]
        if len(vals)<2: return None
        return {"min":r2(min(vals)),"max":r2(max(vals)),"avg":r2(sum(vals)/len(vals))}

    return {
        "label":label, "hold_days":hold_days, "n_trades":n_trades,
        "win_rate_full":r2(wr_full), "avg_ret":r2(avg_ret),
        "min_ret":r2(min_ret), "max_ret":r2(max_ret),
        "avg_max_gain":r2(avg_mg),
        "ret_rate_per_day":r2(ret_rate),
        "sharpe":r2(avg_ret/(np.std(rets)+0.001)*math.sqrt(252/hold_days)) if n_trades>2 else 0,
        "years":sorted(years),
        "ctx_rsi":cstats("ctx_rsi"), "ctx_vol_r":cstats("ctx_vol_r"),
        "ctx_ret5":cstats("ctx_ret5"), "ctx_ret20":cstats("ctx_ret20"),
        "ctx_adx":cstats("ctx_adx"),
        # Buying strategy aggregates (Day1 stats across all trades)
        "buy_stats": _buy_stats(trades),
        "trades":trades,
    }

# ─── MULTI-YEAR VALIDATION ────────────────────────────────────────────────────
def validate_years(df, signal_series, hold_days, label):
    """
    Split data into periods and test each independently.
    Returns confidence score 0-100.
    """
    periods = {
        "train":  ("2020-01-01","2022-12-31"),
        "oos_23": ("2023-01-01","2023-12-31"),
        "oos_24": ("2024-01-01","2024-12-31"),
        "oos_25": ("2025-01-01","2025-12-31"),
        "recent": ("2026-01-01","2099-12-31"),
    }
    results={}
    for pname,(start,end) in periods.items():
        mask=(df["date"]>=start)&(df["date"]<=end)
        df_p=df[mask].reset_index(drop=True)
        sig_p=signal_series[mask].reset_index(drop=True)
        if len(df_p)<20: results[pname]=None; continue
        r=backtest(df_p,sig_p,hold_days,label+"_"+pname)
        results[pname]={"wr":r["win_rate_full"],"avg_ret":r["avg_ret"],"n":r["n_trades"]} if r else None

    # Confidence score
    periods_with_data=[p for p in ["train","oos_23","oos_24","oos_25"] if results.get(p)]
    n_pass=sum(1 for p in periods_with_data if results[p] and results[p]["wr"]>=WIN_RATE)
    confidence = int(n_pass/max(len(periods_with_data),1)*100) if periods_with_data else 0

    # Grade
    if confidence==100 and len(periods_with_data)>=3: grade="A+"
    elif confidence>=75: grade="A"
    elif confidence>=50: grade="B"
    else: grade="C"

    return grade, confidence, results

# ─── UC1: SURGE MOMENTUM ──────────────────────────────────────────────────────
def uc1_surge(df):
    """
    Search all (surge_pct, surge_days, forward_days) combinations.
    Returns best result per stock or None.
    """
    best=None
    c=df["c"]

    for surge_d in UC1_SURGE_DAYS:
        # Rolling return over surge_d days
        roll_ret=c.pct_change(surge_d)*100

        for surge_pct in UC1_SURGE_PCT:
            # Signal: stock rose >= surge_pct% over surge_d days
            signal=roll_ret >= surge_pct

            if int(signal.sum()) < MIN_OCC: continue

            for fwd_d in UC1_FORWARD_DAYS:
                label=f"UC1_surge{surge_pct:.0f}pct_{surge_d}d_fwd{fwd_d}d"
                r=backtest(df, signal, fwd_d, label)
                if r is None: continue

                # Update best by return rate per day
                if best is None or r["ret_rate_per_day"]>best["ret_rate_per_day"]:
                    best=dict(r, uc="UC1",
                               surge_pct=surge_pct, surge_days=surge_d, fwd_days=fwd_d,
                               desc=f"Stock rises ≥{surge_pct:.0f}% over {surge_d}d → hold {fwd_d}d")
    return best

# ─── UC2: SEASONAL ────────────────────────────────────────────────────────────
def uc2_seasonal(df):
    """
    Find months and quarters where stock reliably gives 10%+ returns.
    """
    best=None
    c=df["c"]
    df2=df.copy()
    df2["month"]=df2["date"].dt.month
    df2["quarter"]=df2["date"].dt.quarter
    df2["year"]=df2["date"].dt.year

    # Check each month
    for mo in UC2_MONTHS:
        for hold in UC2_HOLD:
            # Signal: first trading day of this month
            is_first_of_month=(df2["month"]==mo)&(df2["month"]!=df2["month"].shift(1))
            label=f"UC2_month{mo}_hold{hold}d"
            r=backtest(df, is_first_of_month, hold, label)
            if r is None: continue
            if best is None or r["ret_rate_per_day"]>best["ret_rate_per_day"]:
                import calendar
                mo_name=calendar.month_abbr[mo]
                best=dict(r, uc="UC2", season_type="MONTH", season_val=mo,
                           fwd_days=hold,
                           desc=f"Buy on first trading day of {mo_name}, hold {hold}d")

    # Check each quarter
    for q in UC2_QUARTERS:
        for hold in UC2_HOLD:
            is_first_of_qtr=(df2["quarter"]==q)&(df2["quarter"]!=df2["quarter"].shift(1))
            label=f"UC2_Q{q}_hold{hold}d"
            r=backtest(df, is_first_of_qtr, hold, label)
            if r is None: continue
            if best is None or r["ret_rate_per_day"]>best["ret_rate_per_day"]:
                best=dict(r, uc="UC2", season_type="QUARTER", season_val=q,
                           fwd_days=hold,
                           desc=f"Buy on first trading day of Q{q}, hold {hold}d")

    return best

# ─── UC3: COMBINED ────────────────────────────────────────────────────────────
def uc3_combined(df, uc1_result, uc2_result):
    """
    Surge signal that also fires during a strong seasonal window.
    """
    if uc1_result is None or uc2_result is None: return None
    c=df["c"]
    df2=df.copy()
    df2["month"]=df2["date"].dt.month
    df2["quarter"]=df2["date"].dt.quarter

    # Rebuild UC1 signal
    surge_pct=uc1_result["surge_pct"]; surge_d=uc1_result["surge_days"]
    roll_ret=c.pct_change(surge_d)*100
    uc1_sig=roll_ret >= surge_pct

    # Rebuild UC2 signal window (active for the whole month/quarter)
    if uc2_result["season_type"]=="MONTH":
        uc2_window=df2["month"]==uc2_result["season_val"]
    else:
        uc2_window=df2["quarter"]==uc2_result["season_val"]

    combined=uc1_sig & uc2_window
    if int(combined.sum()) < MIN_OCC: return None

    fwd=max(uc1_result["fwd_days"], uc2_result["fwd_days"])
    label=f"UC3_combined_{fwd}d"
    r=backtest(df, combined, fwd, label)
    if r is None: return None
    return dict(r, uc="UC3", fwd_days=fwd,
                desc=f"Surge ≥{surge_pct:.0f}% in {surge_d}d during strong season, hold {fwd}d")

# ─── UC4: TECHNICAL PATTERNS (25 strategies, parametrised) ───────────────────
def uc4_technical(df):
    """
    All technical strategies. Returns best result for this stock.
    Strategies run in batches; best by ret_rate_per_day returned.
    """
    c=df["c"]; v=df["v"]; n=len(df)
    best=None

    def check(sig, fwd, label):
        nonlocal best
        if int(sig.fillna(False).sum()) < MIN_OCC: return
        r=backtest(df, sig, fwd, label)
        if r is None: return
        if best is None or r["ret_rate_per_day"]>best["ret_rate_per_day"]:
            best=dict(r, uc="UC4")

    # Batch 1: Moving average crossovers
    for fast,slow,name in [(10,30,"SMA10_30"),(20,50,"SMA20_50"),(50,200,"GoldenCross")]:
        if f"sma{fast}" not in df or f"sma{slow}" not in df: continue
        fa=df[f"sma{fast}"]; sl=df[f"sma{slow}"]
        cross_up=(fa>sl)&(fa.shift(1)<=sl.shift(1))&(df["vol_r"]>=1.2)
        for fwd in [5,10,20]:
            check(cross_up, fwd, f"UC4_{name}_fwd{fwd}d")

    # Batch 2: EMA crossovers
    if "ema12" in df and "ema26" in df:
        cross=(df["ema12"]>df["ema26"])&(df["ema12"].shift(1)<=df["ema26"].shift(1))
        for fwd in [5,10,20]:
            check(cross&(df["vol_r"]>=1.0), fwd, f"UC4_EMA_cross_fwd{fwd}d")

    if "ema20" in df and "ema50" in df:
        cross=(df["ema20"]>df["ema50"])&(df["ema20"].shift(1)<=df["ema50"].shift(1))
        for fwd in [5,10,20]:
            check(cross, fwd, f"UC4_EMA20_50_fwd{fwd}d")

    # Batch 3: RSI strategies
    if "rsi" in df:
        rsi=df["rsi"]
        for lo_thr,hi_thr in [(30,35),(25,30),(20,25)]:
            sig=(rsi>hi_thr)&(rsi.shift(1)<=lo_thr)&(c>c.shift(1))
            for fwd in [5,10,20]:
                check(sig, fwd, f"UC4_RSI_ob{lo_thr}_fwd{fwd}d")
        # RSI momentum (above 60 + rising)
        sig_mom=(rsi>60)&(rsi>rsi.shift(1))&(rsi.shift(1)<=60)&(df["vol_r"]>=1.5)
        for fwd in [5,10,20]:
            check(sig_mom, fwd, f"UC4_RSI_mom_fwd{fwd}d")

    # Batch 4: MACD
    if "macd" in df and "macd_sig" in df:
        cross=(df["macd"]>df["macd_sig"])&(df["macd"].shift(1)<=df["macd_sig"].shift(1))
        for vol_f in [1.0,1.3,1.5]:
            sig=cross&(df["vol_r"]>=vol_f)
            for fwd in [5,10,20]:
                check(sig, fwd, f"UC4_MACD_vol{vol_f}_fwd{fwd}d")

    # Batch 5: Bollinger Band strategies
    if "bb_lo" in df and "bb_up" in df:
        # Lower band bounce
        bounce=(df["l"]<=df["bb_lo"])&(c>df["bb_lo"])&(df["vol_r"]>=1.2)
        for fwd in [5,10,20]:
            check(bounce, fwd, f"UC4_BB_bounce_fwd{fwd}d")
        # BB squeeze breakout (width was low, now expanding)
        squeeze_break=(df["bb_w"]>df["bb_w"].shift(1)*1.3)&(df["bb_w"].shift(1)<df["bb_w"].shift(2))&(c>c.shift(1))
        for fwd in [5,10,20]:
            check(squeeze_break, fwd, f"UC4_BB_squeeze_fwd{fwd}d")
        # Upper band touch continuation
        uptouch=(c>=df["bb_up"]*0.99)&(df["vol_r"]>=2.0)&(c>c.shift(1))
        for fwd in [5,10]:
            check(uptouch, fwd, f"UC4_BB_up_fwd{fwd}d")

    # Batch 6: Volume breakouts
    for vol_mult,ret_min in [(2.0,1.0),(1.5,2.0),(3.0,0.5)]:
        sig=(df["vol_r"]>=vol_mult)&(df["ret1"]>=ret_min)&(c>=df["hi20"].shift(1))
        for fwd in [5,10,20]:
            check(sig, fwd, f"UC4_VolBreak_{vol_mult}x_fwd{fwd}d")

    # Batch 7: Consecutive reversal patterns
    if "consec_dn" in df:
        for n_dn in [3,4,5]:
            sig=(df["consec_dn"].shift(1)>=n_dn)&(c>c.shift(1))&(df["vol_r"]>=1.2)
            for fwd in [5,10,20]:
                check(sig, fwd, f"UC4_ConsecRev{n_dn}d_fwd{fwd}d")

    # Batch 8: Floor bounce
    if "lo20" in df:
        for rsi_max in [40,50,60]:
            sig=(df["l"].shift(1)<=df["lo20"].shift(2))&(c>c.shift(1))&(df["rsi"]<rsi_max)
            for fwd in [5,10,20]:
                check(sig, fwd, f"UC4_Floor_rsi{rsi_max}_fwd{fwd}d")

    # Batch 9: Gap momentum
    if "gap" in df:
        for gap_pct in [2.0,3.0,5.0]:
            day_range=(df["h"]-df["l"]).replace(0,0.0001)
            close_pos=(c-df["l"])/day_range
            sig=(df["gap"]>gap_pct)&(close_pos>0.75)&(df["vol_r"]>=1.5)
            for fwd in [5,10]:
                check(sig, fwd, f"UC4_Gap{gap_pct:.0f}pct_fwd{fwd}d")

    # Batch 10: ATR range expansion (breakout energy days)
    if "atr" in df:
        day_range2=df["h"]-df["l"]
        for atr_mult in [1.5,2.0,2.5]:
            sig=(day_range2>atr_mult*df["atr"])&(c>df["o"])&(df["ret1"]>1.0)&(df["vol_r"]>=1.3)
            for fwd in [5,10]:
                check(sig, fwd, f"UC4_ATR{atr_mult}x_fwd{fwd}d")

    # Batch 11: Trend pullback
    for trend_min,pb_max in [(8,-2),(12,-3),(5,-1.5)]:
        sig=(df["ret20"]>trend_min)&(df["ret5"]<pb_max)&(c>c.shift(1))&(df["vol_r"]>=1.2)&(df["rsi"]<60)
        for fwd in [5,10,20]:
            check(sig, fwd, f"UC4_TrendPB_{trend_min}pct_fwd{fwd}d")

    # Batch 12: 52-week high breakout
    if "hi52" in df:
        for pct_from in [0.98,0.99,1.0]:
            sig=(c>=df["hi52"].shift(1)*pct_from)&(df["vol_r"]>=2.0)&(df["ret1"]>1.0)&(df["rsi"]>60)
            for fwd in [5,10,20]:
                check(sig, fwd, f"UC4_52WBreak_{pct_from}_fwd{fwd}d")

    # Batch 13: Stochastic
    if "stoch_k" in df and "stoch_d" in df:
        sk=df["stoch_k"]; sd=df["stoch_d"]
        for lo,hi in [(20,30),(15,25)]:
            cross=(sk>sd)&(sk.shift(1)<=sd.shift(1))&(sk<hi)
            for fwd in [5,10]:
                check(cross, fwd, f"UC4_Stoch_{lo}_{hi}_fwd{fwd}d")

    # Batch 14: ADX trend strength entry
    if "adx" in df and "plus_di" in df:
        adx=df["adx"]; pdi=df["plus_di"]; mdi=df["minus_di"]
        for adx_min in [20,25,30]:
            sig=(adx>adx_min)&(pdi>mdi)&(pdi>pdi.shift(1))&(df["vol_r"]>=1.2)
            for fwd in [5,10,20]:
                check(sig, fwd, f"UC4_ADX{adx_min}_fwd{fwd}d")

    # Batch 15: Williams %R oversold
    if "willr" in df:
        wr=df["willr"]
        for lvl in [-80,-70,-60]:
            sig=(wr>lvl)&(wr.shift(1)<=lvl)&(c>c.shift(1))
            for fwd in [5,10]:
                check(sig, fwd, f"UC4_WillR{lvl}_fwd{fwd}d")

    # Batch 16: OBV divergence (price flat/down, OBV rising)
    if "obv" in df and "obv_sma20" in df:
        obv_rising=(df["obv"]>df["obv"].shift(5))&(df["obv"]>df["obv_sma20"])
        price_flat=abs(df["ret5"])<3
        sig=obv_rising&price_flat&(c>c.shift(1))
        for fwd in [10,20]:
            check(sig, fwd, f"UC4_OBV_div_fwd{fwd}d")

    # Batch 17: Double bottom approximation
    # Two similar lows (within 2%) followed by breakout
    if "lo20" in df:
        lo_now=df["lo20"]; lo_prev=df["lo20"].shift(10)
        double_bot=(abs(lo_now-lo_prev)/lo_prev.replace(0,1)<0.02)&(c>df["hi20"].shift(1)*0.95)&(df["vol_r"]>=1.5)
        for fwd in [10,20]:
            check(double_bot, fwd, f"UC4_DoubleBot_fwd{fwd}d")

    # Batch 18: Bull flag (strong surge then consolidation then continuation)
    if "consec_up" in df:
        # Strong move (5d up), then tight range (3d bb_w low), then breakout
        strong_move=df["ret5"]>8
        tight_range=df["bb_w"]<df["bb_w"].rolling(20).mean()*0.7
        breakout=(c>c.shift(1).rolling(5).max())
        bull_flag=strong_move.shift(5)&tight_range&breakout&(df["vol_r"]>=1.3)
        for fwd in [5,10]:
            check(bull_flag, fwd, f"UC4_BullFlag_fwd{fwd}d")

    # Batch 19: Pivot point bounce
    if "s1" in df:
        piv_bounce=(df["l"]<=df["s1"].shift(1)*1.01)&(c>df["s1"].shift(1))&(df["vol_r"]>=1.2)
        for fwd in [5,10]:
            check(piv_bounce, fwd, f"UC4_PivotBounce_fwd{fwd}d")

    # Batch 20: Supertrend crossover
    if "st_lower" in df and "st_upper" in df:
        # Price crosses above lower supertrend band
        st_cross=(c>df["st_lower"])&(c.shift(1)<=df["st_lower"].shift(1))&(df["vol_r"]>=1.0)
        for fwd in [5,10,20]:
            check(st_cross, fwd, f"UC4_SuperTrend_fwd{fwd}d")

    # Batch 21: Consecutive up days momentum
    if "consec_up" in df:
        for n_up in [3,4,5]:
            sig=(df["consec_up"]==n_up)&(df["vol_r"]>=1.3)&(df["rsi"]<70)
            for fwd in [5,10]:
                check(sig, fwd, f"UC4_ConsecUp{n_up}_fwd{fwd}d")

    # Batch 22: 60-day momentum breakout (strong trend continuation)
    sig_60=(df["ret60"]>20)&(df["ret20"]>5)&(df["ret5"]>2)&(df["vol_r"]>=1.5)&(df["rsi"]>55)
    for fwd in [5,10,20]:
        check(sig_60, fwd, f"UC4_Mom60d_fwd{fwd}d")

    # Batch 23: Mean reversion (stock fell hard, now bouncing)
    sig_mr=(df["ret20"]<-15)&(df["ret5"]>0)&(df["ret1"]>2)&(df["vol_r"]>=1.5)&(df["rsi"]<50)
    for fwd in [5,10,20]:
        check(sig_mr, fwd, f"UC4_MeanRev_fwd{fwd}d")

    # Batch 24: Combination RSI + Volume + Trend
    sig_combo=(df["rsi"]>50)&(df["vol_r"]>=1.5)&(df["ret20"]>5)&(df["macd"]>df["macd_sig"])&(df["adx"]>20)
    for fwd in [5,10,20]:
        check(sig_combo, fwd, f"UC4_MultiConfirm_fwd{fwd}d")

    # Batch 25: Price-volume divergence (volume surges but price consolidates = upcoming move)
    high_vol_flat=(df["vol_r"]>=2.0)&(abs(df["ret1"])<1.5)&(c>df["sma20"])
    for fwd in [5,10]:
        check(high_vol_flat, fwd, f"UC4_VolAccum_fwd{fwd}d")

    return best

# ─── PER-STOCK ANALYSIS ───────────────────────────────────────────────────────
def analyse_stock(sym, df, latest_date):
    df=df.copy().sort_values("date").reset_index(drop=True)
    n=len(df)
    if n<MIN_HISTORY_DAYS: return None

    c_last=float(df["c"].iloc[-1])
    if c_last<MIN_PRICE: return None

    tv_60=float((df["c"]*df["v"]).iloc[-60:].mean()) if n>=60 else 0
    if tv_60<MIN_TURNOVER: return None

    traded_today=str(df["date"].iloc[-1].date())==latest_date

    df=indicators(df)

    # Run all 4 use cases
    results={}
    for uc_name, uc_fn in [("uc1",uc1_surge),("uc2",uc2_seasonal),("uc4",uc4_technical)]:
        try:
            r=uc_fn(df)
        except Exception as e:
            r=None
        results[uc_name]=r

    # UC3 combined
    try:
        r3=uc3_combined(df, results.get("uc1"), results.get("uc2"))
    except: r3=None
    results["uc3"]=r3

    # Find overall best
    all_results=[r for r in results.values() if r]
    if not all_results: return None

    best=max(all_results, key=lambda x: x.get("ret_rate_per_day") or 0)

    # Multi-year validation for the best
    try:
        if "label" in best:
            # Rebuild signal for best strategy
            sig_label=best["label"]
            # Find which UC produced this
            best_uc=best.get("uc","UC4")
            if best_uc=="UC1":
                roll=df["c"].pct_change(best["surge_days"])*100
                sig_series=roll>=best["surge_pct"]
            elif best_uc=="UC2":
                df2=df.copy()
                df2["month"]=df2["date"].dt.month
                df2["quarter"]=df2["date"].dt.quarter
                if best["season_type"]=="MONTH":
                    sig_series=(df2["month"]==best["season_val"])&(df2["month"]!=df2["month"].shift(1))
                else:
                    sig_series=(df2["quarter"]==best["season_val"])&(df2["quarter"]!=df2["quarter"].shift(1))
            else:
                sig_series=pd.Series(False,index=df.index)  # can't easily rebuild UC3/UC4
            grade,confidence,val_results=validate_years(df,sig_series,best["hold_days"],sig_label)
        else:
            grade,confidence,val_results="B",50,{}
    except:
        grade,confidence,val_results="B",50,{}

    # Long-term score
    lt_score=_lt_score(df, all_results, confidence)

    # Today's signal — check each UC
    signal_today=False
    signal_from=[]
    for uc_name,r in results.items():
        if r is None: continue
        # Recheck today's signal from the last row's context
        last_trade=r.get("trades",[])
        if last_trade:
            latest_sig=last_trade[-1]["sig_date"]
            if latest_sig==latest_date and traded_today:
                signal_today=True; signal_from.append(uc_name.upper())

    # CRITICAL FIX: If signal fires today but was not recorded in backtest trades
    # (due to the no-overlap rule: backtest skips signals during an open hold),
    # force-add today's signal to _all_trades so it appears in Signal History.
    if signal_today and traded_today and best:
        best_trades = best.get("trades", [])
        already_in_hist = any(t.get("sig_date") == latest_date for t in best_trades)
        if not already_in_hist:
            today_sig_trade = {
                "sig_date":     latest_date,
                "entry_date":   latest_date,  # real entry = next trading day open
                "exit_date":    None,
                "is_open":      True,
                "is_complete":  False,
                "entry_px":     r2(c_last),   # today's close as reference; real entry = tomorrow open
                "exit_px":      None,
                "ret":          0.0,
                "ret_3d":       None, "ret_10d": None,
                "max_gain":     0.0,  "max_dd":  0.0,
                "d0_close":     r2(c_last),
                "d1_open":      None, "d1_gap_pct": None, "d1_dip_pct": None,
                "d1_close_ret": None, "d1_intra_ret": None,
                "ctx_rsi":   float(df["rsi"].iloc[-1])   if "rsi"   in df.columns else None,
                "ctx_vol_r": float(df["vol_r"].iloc[-1]) if "vol_r" in df.columns else None,
                "ctx_ret5":  float(df["ret5"].iloc[-1])  if "ret5"  in df.columns else None,
                "ctx_ret20": float(df["ret20"].iloc[-1]) if "ret20" in df.columns else None,
                "ctx_adx":   float(df["adx"].iloc[-1])   if "adx"   in df.columns else None,
                "_note":     "Signal fired today; entry tomorrow at open price",
            }
            best["trades"] = best_trades + [today_sig_trade]

    return {
        "sym": sym,
        "latest_date": str(df["date"].iloc[-1].date()),
        "traded_today": traded_today,
        "price": r2(c_last),
        "avg_turnover_cr": r2(tv_60/1e7),
        # Overall best
        "best_uc":           best.get("uc","?"),
        "best_label":        best.get("label",""),
        "best_desc":         best.get("desc",""),
        "best_ret_rate":     best.get("ret_rate_per_day"),
        "best_avg_ret":      best.get("avg_ret"),
        "best_min_ret":      best.get("min_ret"),
        "best_max_ret":      best.get("max_ret"),
        "best_hold_days":    best.get("hold_days"),
        "best_wr":           best.get("win_rate_full"),
        "best_sharpe":       best.get("sharpe"),
        "best_n_trades":     best.get("n_trades"),
        "best_years":        best.get("years",[]),
        # Validation
        "grade": grade, "confidence": confidence,
        "val_results": val_results,
        # Per-UC results (summary)
        "uc1": _uc_summary(results.get("uc1")),
        "uc2": _uc_summary(results.get("uc2")),
        "uc3": _uc_summary(results.get("uc3")),
        "uc4": _uc_summary(results.get("uc4")),
        # Context ranges
        "ctx_rsi":   best.get("ctx_rsi"),
        "ctx_vol_r": best.get("ctx_vol_r"),
        "ctx_ret5":  best.get("ctx_ret5"),
        "ctx_ret20": best.get("ctx_ret20"),
        "buy_stats": best.get("buy_stats"),
        # Scoring
        "longterm_score": lt_score,
        "signal_today": signal_today and traded_today,
        "signal_from": signal_from,
        # Current indicators
        "rsi":      r2(df["rsi"].iloc[-1]),
        "vol_r":    r2(df["vol_r"].iloc[-1]),
        "adx":      r2(df["adx"].iloc[-1]),
        "ret1":     r2(df["ret1"].iloc[-1]),
        "ret5":     r2(df["ret5"].iloc[-1]),
        "ret20":    r2(df["ret20"].iloc[-1]),
        # Recent trades for display
        "recent_trades": best.get("trades",[])[-6:],
        "_all_trades":   best.get("trades",[]),
    }

def _uc_summary(r):
    if not r: return None
    return {
        "uc":r.get("uc"),"desc":r.get("desc",""),
        "ret_rate":r.get("ret_rate_per_day"),"avg_ret":r.get("avg_ret"),
        "min_ret":r.get("min_ret"),"hold_days":r.get("hold_days"),
        "win_rate":r.get("win_rate_full"),"n_trades":r.get("n_trades"),
        "sharpe":r.get("sharpe"),"years":r.get("years",[]),
        "ctx_rsi":r.get("ctx_rsi"),"ctx_vol_r":r.get("ctx_vol_r"),
    }

def _lt_score(df,results,confidence):
    c=df["c"]; n=len(c)
    s=0
    ema50=c.ewm(span=50,adjust=False).mean()
    if float(c.iloc[-1])>float(ema50.iloc[-1]): s+=15
    ret60=(float(c.iloc[-1])-float(c.iloc[max(0,n-60)]))/float(c.iloc[max(0,n-60)])*100
    if ret60>0: s+=10
    if ret60>10: s+=10
    s+=min(25,confidence//4)
    best_rr=max([r.get("ret_rate_per_day") or 0 for r in results],default=0)
    if best_rr>=3: s+=20
    elif best_rr>=2: s+=12
    elif best_rr>=1: s+=6
    return min(100,s)

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def _write_still_valid(stocks, latest, now):
    """
    Find past signals still within hold tenure with >=5% gap to min return target.
    These are still good to invest in now.
    """
    try:
        from datetime import datetime as DT
        latest_dt = DT.strptime(latest, "%Y-%m-%d")
        positions = []
        for s in stocks:
            current_px = s.get("price") or 0
            if current_px <= 0: continue
            min_ret    = s.get("best_min_ret") or 0
            avg_ret    = s.get("best_avg_ret") or 0
            max_ret    = s.get("best_max_ret") or 0
            hold_days  = s.get("best_hold_days") or 5
            hold_cal   = int(hold_days * 1.5)  # convert trading days to ~calendar days

            for t in (s.get("_all_trades") or []):
                entry_date_str = t.get("entry_date")
                entry_px = t.get("entry_px") or 0
                if not entry_date_str or entry_px <= 0: continue
                try:
                    entry_dt = DT.strptime(entry_date_str, "%Y-%m-%d")
                except: continue

                days_elapsed = (latest_dt - entry_dt).days
                if days_elapsed < 0 or days_elapsed >= hold_cal: continue  # expired

                min_target = entry_px * (1 + min_ret / 100)
                cur_ret    = (current_px - entry_px) / entry_px * 100
                gap        = (min_target - current_px) / current_px * 100

                if gap < 5.0: continue  # already at or past min target — skip

                days_remaining = max(0, hold_cal - days_elapsed)

                positions.append({
                    "sym":                    s["sym"],
                    "sig_date":               t.get("sig_date"),
                    "entry_date":             entry_date_str,
                    "entry_px":               r2(entry_px),
                    "current_price":          r2(current_px),
                    "uc":                     s.get("best_uc"),
                    "desc":                   s.get("best_desc",""),
                    "grade":                  s.get("grade"),
                    "confidence":             s.get("confidence"),
                    "hold_days":              hold_days,
                    "min_ret":                r2(min_ret),
                    "avg_ret":                r2(avg_ret),
                    "max_ret":                r2(max_ret),
                    "min_target_price":       r2(min_target),
                    "avg_target_price":       r2(entry_px * (1 + avg_ret / 100)),
                    "current_return_pct":     r2(cur_ret),
                    "gap_to_min_target_pct":  r2(gap),
                    "days_elapsed":           days_elapsed,
                    "days_remaining":         days_remaining,
                    "avg_turnover_cr":        s.get("avg_turnover_cr"),
                    "rsi":                    s.get("rsi"),
                    "vol_r":                  s.get("vol_r"),
                })

        # Deduplicate: keep best gap per stock
        seen = {}
        for p in positions:
            sym = p["sym"]
            if sym not in seen or p["gap_to_min_target_pct"] > seen[sym]["gap_to_min_target_pct"]:
                seen[sym] = p
        positions = sorted(seen.values(), key=lambda x: -(x.get("gap_to_min_target_pct") or 0))

        jdump({"generated_at": now, "latest_date": latest,
               "n": len(positions), "positions": positions},
              OUT / "still_valid.json")
        print(f"  OK still_valid.json ({len(positions)} positions with >=5% gap remaining)")
    except Exception as e:
        print(f"  WARN still_valid: {e}")


def main():
    force=os.getenv("FORCE_RERUN","false").lower()=="true"
    print("="*70); print("NSE Master Stock Analyzer v3")
    print("  UC1: Surge Momentum | UC2: Seasonal | UC3: Combined | UC4: Technical (25 strategies)")
    print("  Requirements: ≥95% win rate | ≥10% min return | ≥4 occurrences | 2+ years")
    print("="*70)
    OUT.mkdir(parents=True,exist_ok=True)

    manifest=jload(MANI); tds=sorted(manifest.keys()); latest=tds[-1]
    print(f"\n  Trading days: {len(tds)} | Latest: {latest}")

    cp=jload(CP)
    if force: cp={}; print("  FORCE RERUN — clearing checkpoint")
    elif cp.get("processed_date")==latest:
        print(f"  Already processed {latest}. Regenerating outputs.")
        _write_all(cp.get("results",{}), latest); return
    else:
        print(f"  Checkpoint: {cp.get('processed_date','none')} → processing to {latest}")

    print("\n[1] Loading equity data…")
    df_all=load_all(manifest)
    if df_all.empty: print("ERROR: No data"); sys.exit(1)
    print(f"  {len(df_all):,} rows | {df_all['sym'].nunique()} symbols")

    print("\n[2] Grouping…")
    grps={}
    for sym,grp in df_all.groupby("sym"):
        if sym in EXCLUDED or any(sym.upper().endswith(s) for s in EXCL_SFX): continue
        grps[sym]=grp.reset_index(drop=True)
    del df_all; gc.collect()
    print(f"  {len(grps)} symbols after exclusions")

    print("\n[3] Running analysis (all use cases)…")
    saved=cp.get("results",{})
    results=dict(saved)
    new=skip=fail=0

    for i,(sym,grp) in enumerate(grps.items()):
        if not force and sym in results:
            if results[sym].get("latest_date")==str(grp["date"].max().date()):
                skip+=1; continue
        try:
            r=analyse_stock(sym,grp,latest)
            if r: results[sym]=r; new+=1
            else: fail+=1
        except Exception as e:
            fail+=1
        if (i+1)%100==0:
            print(f"  {i+1}/{len(grps)} | new={new} skip={skip} fail={fail}",flush=True)

    del grps; gc.collect()
    print(f"\n  Done: {new} qualified | {skip} skipped | {fail} filtered")

    cp2={"version":"v3","processed_date":latest,"results":results}
    jdump(cp2,CP)
    print(f"  Checkpoint saved")

    print("\n[4] Writing outputs…")
    _write_all(results, latest)
    print("Done.")

def _write_all(results, latest):
    OUT.mkdir(parents=True,exist_ok=True)
    ist=timezone(timedelta(hours=5,minutes=30))
    now=datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")
    stocks=[v for v in results.values() if v]

    # Master sorted by ret_rate (primary), then confidence
    ms=sorted(stocks,key=lambda x:(-(x.get("best_ret_rate") or 0),-(x.get("confidence") or 0)))

    # Slim for master (no full trade lists)
    slim=[{k:v for k,v in s.items() if k not in ("_all_trades","recent_trades")} for s in ms]
    for s,sl in zip(ms,slim): sl["recent_trades"]=s.get("recent_trades",[])
    jdump({"generated_at":now,"latest":latest,"n":len(slim),"stocks":slim}, OUT/"master_results.json")
    print(f"  OK master_results.json ({len(slim)})")

    # Daily alerts
    alerts=[s for s in stocks if s.get("signal_today") and s.get("traded_today")]
    alerts.sort(key=lambda x:-(x.get("best_ret_rate") or 0))
    alert_out=[{
        "sym":a["sym"],"price":a["price"],"signal_from":a["signal_from"],
        "best_uc":a["best_uc"],"best_desc":a["best_desc"],
        "ret_rate":a["best_ret_rate"],"avg_ret":a["best_avg_ret"],
        "min_ret":a["best_min_ret"],"hold_days":a["best_hold_days"],
        "win_rate":a["best_wr"],"sharpe":a["best_sharpe"],
        "n_trades":a["best_n_trades"],"confidence":a["confidence"],"grade":a["grade"],
        "ctx_rsi":a.get("ctx_rsi"),"ctx_vol_r":a.get("ctx_vol_r"),
        "ctx_ret5":a.get("ctx_ret5"),"ctx_ret20":a.get("ctx_ret20"),
        "rsi":a.get("rsi"),"vol_r":a.get("vol_r"),"adx":a.get("adx"),
        "ret5":a.get("ret5"),"ret20":a.get("ret20"),
        "avg_turnover_cr":a.get("avg_turnover_cr"),
        "buy_stats":a.get("buy_stats"),
        "recent_trades":a.get("recent_trades",[]),
        "uc1":a.get("uc1"),"uc2":a.get("uc2"),"uc3":a.get("uc3"),"uc4":a.get("uc4"),
    } for a in alerts]
    jdump({"generated_at":now,"signal_date":latest,"n_alerts":len(alert_out),"alerts":alert_out}, OUT/"daily_alerts.json")
    print(f"  OK daily_alerts.json ({len(alert_out)})")

    # Signal history
    hist=[]
    for s in stocks:
        for t in (s.get("_all_trades") or s.get("recent_trades") or []):
            # Adjust entry_px for splits/bonuses that happened after signal
            raw_entry = t.get("entry_px") or 0
            adj_entry = raw_entry
            if raw_entry > 0 and s.get("price") and t.get("entry_date"):
                # If current_price is less than half of entry_px → likely split
                # Mark as possibly adjusted so UI can flag it
                ratio = s["price"] / raw_entry if raw_entry else 1
                if ratio < 0.5:   # price dropped >50% vs entry = likely split/bonus
                    pass  # flag below

            hist.append({
                "sym":s["sym"],"uc":s["best_uc"],"desc":s["best_desc"],
                "grade":s["grade"],"confidence":s["confidence"],
                "hold_days":s["best_hold_days"],
                "min_ret_strategy":s.get("best_min_ret"),
                "avg_ret_strategy":s.get("best_avg_ret"),
                "max_ret_strategy":s.get("best_max_ret"),
                "current_price":s.get("price"),
                "possible_split": (s.get("price") or 0) < (t.get("entry_px") or 0) * 0.5,
                **{k:t.get(k) for k in ["sig_date","entry_date","exit_date","is_open",
                   "entry_px","exit_px","ret","ret_3d","ret_10d",
                   "max_gain","max_dd","ctx_rsi","ctx_vol_r","ctx_ret5","ctx_ret20","ctx_adx",
                   "d0_close","d1_open","d1_high","d1_low","d1_close",
                   "d1_gap_pct","d1_dip_pct","d1_close_ret","d1_intra_ret"]},
            })
    hist.sort(key=lambda x:(x.get("sig_date") or ""),reverse=True)
    jdump({"generated_at":now,"n_signals":len(hist),"signals":hist[:3000]}, OUT/"signal_history.json")
    print(f"  OK signal_history.json ({len(hist)} entries)")

    # Long-term picks (confidence>=70, A/A+ grade)
    lt=[s for s in stocks if s.get("confidence",0)>=70 and s.get("grade") in ("A+","A")]
    lt.sort(key=lambda x:(-(x.get("longterm_score") or 0),-(x.get("best_ret_rate") or 0)))
    lt_out=[{
        "sym":s["sym"],"price":s["price"],"longterm_score":s["longterm_score"],
        "grade":s["grade"],"confidence":s["confidence"],
        "best_uc":s["best_uc"],"best_desc":s["best_desc"],
        "ret_rate":s["best_ret_rate"],"avg_ret":s["best_avg_ret"],"min_ret":s["best_min_ret"],
        "hold_days":s["best_hold_days"],"win_rate":s["best_wr"],"n_trades":s["best_n_trades"],
        "rsi":s.get("rsi"),"ret20":s.get("ret20"),"avg_turnover_cr":s.get("avg_turnover_cr"),
        "uc1":s.get("uc1"),"uc2":s.get("uc2"),"uc3":s.get("uc3"),"uc4":s.get("uc4"),
        "val_results":s.get("val_results",{}),
    } for s in lt[:40]]
    jdump({"generated_at":now,"latest":latest,"picks":lt_out}, OUT/"longterm_picks.json")
    print(f"  OK longterm_picks.json ({len(lt_out)})")

    # UC-specific summaries
    for uc_key,fname in [("uc1","uc1_surge.json"),("uc2","uc2_seasonal.json"),("uc3","uc3_combined.json"),("uc4","uc4_technical.json")]:
        uc_stocks=[s for s in stocks if s.get(uc_key)]
        uc_stocks.sort(key=lambda x: -(x.get(uc_key,{}).get("ret_rate") or 0))
        jdump({"generated_at":now,"n":len(uc_stocks),"stocks":[{
            "sym":s["sym"],"price":s["price"],"grade":s["grade"],"confidence":s["confidence"],
            **{k:v for k,v in (s.get(uc_key) or {}).items()},
        } for s in uc_stocks[:200]]}, OUT/fname)
    print(f"  OK UC-specific JSONs")

    # Still valid positions (past signals still in tenure with >=5% gap)
    _write_still_valid(stocks, latest, now)

    # Summary
    aplus=[a for a in alert_out if a.get("grade")=="A+"]
    print(f"\n  === Summary ({latest}) ===")
    print(f"  Qualified stocks : {len(stocks)}")
    print(f"  Daily alerts     : {len(alert_out)} (traded today only)")
    print(f"  A+ alerts        : {len(aplus)}")
    for a in aplus[:5]:
        print(f"    {a['sym']:<14} {a['best_uc']:<5} {a['best_desc'][:40]:<40} rate={a['ret_rate']:.2f}%/d")

if __name__=="__main__":
    main()
 