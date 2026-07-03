#!/usr/bin/env python3
"""
breakout_analyzer.py — NSE Consolidation Breakout Detector
Decision-making system built from empirical pattern analysis of 200+ breakouts.

KEY FINDINGS BAKED INTO THIS CODE:
1. Shape fingerprints (4-day trajectory U/F/D) predict outcome better than any single metric
2. D1 flat/dip → D2 ALWAYS positive (28/28 in data) — best dip-buy signal
3. D1 strong (>3%) → D2 is only 50/50 — don't chase
4. Low brk% (2-4%) = late movers (profit on D3/D4), high brk% (>6%) = early movers (profit D1/D2)
5. D4 > D3 > 0 = still accelerating → HOLD BEYOND T+4 (T+5/6 likely positive)
6. Best entry: T0 close or T1 close (not T2 — waiting costs avg 7% in returns)

SHAPE HIT RATES (10%+ peak):
  UUUU/UUUF/UUUD/UDUU/UFUU/FUUF = 100%
  FUUU = 80%,  DUUD = 67%
  FUDU/DUUF/DUFF/DUDU = 0% — AVOID
"""

import json, sys, warnings, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "equity"
OUT  = ROOT / "stock_analysis"
OUT.mkdir(exist_ok=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
MIN_PRICE           = 15.0
MIN_TURNOVER        = 5_000_000
MIN_YEARS           = 2
RECENT_DAYS         = 7
CONSOL_WINDOWS      = [15, 20, 30, 40, 60, 80, 120]
CONSOL_BAND_MAX     = 12.0
BREAKOUT_MIN        = 1.5
MIN_POST_POSITIVE   = 3
POST_DAYS           = 4
MIN_OCCURRENCES     = 2
ALERT_WINDOW        = 5
MIN_GUARANTEED_PEAK = 5.0
PEAK_QUALIFY_RATE   = 0.75

EXCL_CONTAINS = (
    "NIFTY","SENSEX","LIQUID","GILT","SETF","BEES","CPSE",
    "MAFANG","DJML","MONQ","MOSMALL","MASPTOP","HDFCMID",
    "HDFCSML","SMALLCAP","MIDSMALL","BANKPSU","PSUBANK",
    "MOM100","MON100","HDFCNIFBAN","BANKNIFTY",
)

IST     = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)
now_str = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
today   = now_ist.strftime("%Y-%m-%d")

# ── SHAPE STATS (empirically derived from 200+ breakout analysis) ──────────────
# hit_rate = fraction reaching 10%+ peak, avg_pk = average peak %
SHAPE_STATS = {
    'UUUU': {'hit_rate':1.00,'avg_pk':23.1,'n':4, 'grade':'A+','note':'Perfect climb — 100%'},
    'UUUF': {'hit_rate':1.00,'avg_pk':21.9,'n':5, 'grade':'A+','note':'Up 3 days, flat end — 100%'},
    'UUUD': {'hit_rate':1.00,'avg_pk':24.0,'n':6, 'grade':'A+','note':'Up 3, small pullback — 100%'},
    'UDUU': {'hit_rate':1.00,'avg_pk':27.6,'n':3, 'grade':'A+','note':'Up, dip, up, up — 100% avg+28%'},
    'UFUU': {'hit_rate':1.00,'avg_pk':16.1,'n':3, 'grade':'A+','note':'Up, flat, up, up — 100%'},
    'FUUF': {'hit_rate':1.00,'avg_pk':12.7,'n':3, 'grade':'A+','note':'Flat, up, up, flat — 100%'},
    'FUUU': {'hit_rate':0.80,'avg_pk':14.2,'n':5, 'grade':'A', 'note':'Flat start, 3 up — 80%'},
    'DUUD': {'hit_rate':0.67,'avg_pk':10.9,'n':6, 'grade':'B', 'note':'Dip, up, up, dip — 67%'},
    'DUUU': {'hit_rate':0.17,'avg_pk':6.6, 'n':6, 'grade':'D', 'note':'Deep dip start — only 17%'},
    'FFUU': {'hit_rate':1.00,'avg_pk':12.4,'n':1, 'grade':'A', 'note':'Both flat, then up — strong'},
    'DFUU': {'hit_rate':1.00,'avg_pk':17.1,'n':1, 'grade':'A', 'note':'Dip, flat, then rips'},
    'FUDU': {'hit_rate':0.00,'avg_pk':6.9, 'n':3, 'grade':'F', 'note':'AVOID — 0% hit rate'},
    'DUUF': {'hit_rate':0.00,'avg_pk':6.3, 'n':3, 'grade':'F', 'note':'AVOID — 0% hit rate'},
    'DUFF': {'hit_rate':0.00,'avg_pk':3.8, 'n':2, 'grade':'F', 'note':'AVOID — 0% hit rate'},
    'DUDU': {'hit_rate':0.00,'avg_pk':6.8, 'n':2, 'grade':'F', 'note':'AVOID — zigzag failure'},
    'FFFF': {'hit_rate':0.00,'avg_pk':1.1, 'n':1, 'grade':'F', 'note':'AVOID — dead money'},
}

# D1 → D2 prediction (empirically derived)
D1_D2_PREDICTION = {
    'deep_dip':  {'d2_pos_rate':1.00,'n':9, 'note':'D1 deep dip (<-3%) → D2 ALWAYS positive (9/9)'},
    'small_dip': {'d2_pos_rate':1.00,'n':13,'note':'D1 small dip (-3% to -1%) → D2 ALWAYS positive (13/13)'},
    'flat':      {'d2_pos_rate':1.00,'n':15,'note':'D1 flat (-1% to +1%) → D2 ALWAYS positive (15/15)'},
    'small_up':  {'d2_pos_rate':0.78,'n':9, 'note':'D1 small up (1-3%) → D2 positive 78% of time'},
    'strong_up': {'d2_pos_rate':0.50,'n':22,'note':'D1 strong up (>3%) → D2 is 50/50 — do not chase'},
}


def r2(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 2)
    except:
        return None


def encode_day(v):
    """Encode a single daily move as U (up>1%), F (flat), D (down<-1%)"""
    if v is None: return '?'
    if v >  1.0:  return 'U'
    if v < -1.0:  return 'D'
    return 'F'


def daily_moves(post_rets):
    """Convert cumulative T+1..T+4 returns to day-over-day moves D1..D4"""
    t1 = post_rets.get(1)
    t2 = post_rets.get(2)
    t3 = post_rets.get(3)
    t4 = post_rets.get(4)
    d1 = r2(t1)
    d2 = r2(t2 - t1) if t2 is not None and t1 is not None else None
    d3 = r2(t3 - t2) if t3 is not None and t2 is not None else None
    d4 = r2(t4 - t3) if t4 is not None and t3 is not None else None
    return d1, d2, d3, d4


def shape_fingerprint(d1, d2, d3, d4):
    return encode_day(d1) + encode_day(d2) + encode_day(d3) + encode_day(d4)


def d1_category(d1):
    if d1 is None:    return None
    if d1 < -3.0:     return 'deep_dip'
    if d1 < -1.0:     return 'small_dip'
    if d1 <= 1.0:     return 'flat'
    if d1 <= 3.0:     return 'small_up'
    return 'strong_up'


def timing_category(brk_pct):
    """Predict WHEN in the 4-day window the move is likely to happen"""
    if brk_pct is None:  return 'standard'
    if brk_pct < 4.0:    return 'late_mover'   # D3/D4 strongest
    if brk_pct > 9.0:    return 'early_mover'  # D1/D2 strongest
    return 'standard'


def entry_decision(d1, d2, d3, d4, brk_pct, avail_days, brk_close):
    """
    Staged decision system. Returns dict with:
      action, grade, detail, entry_timing, stop_loss, hold_signal
    """
    stop = r2(brk_close * 0.98) if brk_close else None
    timing = timing_category(brk_pct)
    d1c = d1_category(d1)

    # ── T=0: breakout just happened, no post-data ──────────────────────────────
    if avail_days == 0:
        if timing == 'late_mover':
            return {
                'action': 'WATCH',
                'grade': 'C',
                'detail': f'Late mover (brk={brk_pct:.1f}%<4%). Profit expected on D3/D4. '
                          'Do NOT chase at open. Wait for T+1 dip or T+2 consolidation.',
                'entry_timing': 'T+1 dip  OR  T+2 open',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'Usually flat or slight dip — perfect dip-buy setup',
            }
        elif timing == 'early_mover':
            return {
                'action': 'WATCH_FAST',
                'grade': 'B',
                'detail': f'Early mover (brk={brk_pct:.1f}%>9%). D1 likely large. '
                          'Enter at T+1 open if gap-up is <5%, else wait for intraday dip.',
                'entry_timing': 'T+1 open  (if gap <5%)',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'Expect strong D1 — profit in D1/D2 window',
            }
        else:
            return {
                'action': 'WATCH',
                'grade': 'B',
                'detail': f'Standard breakout (brk={brk_pct:.1f}%). '
                          'Enter at T+1 close if D1 is flat/dip. '
                          'If D1 strong (>3%) wait for T+2 dip.',
                'entry_timing': 'T+1 close  (confirm D1 first)',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'D2 is always positive when D1 is flat or dip',
            }

    # ── T=1: D1 known ──────────────────────────────────────────────────────────
    if avail_days == 1 and d1 is not None:
        pred = D1_D2_PREDICTION.get(d1c, {})
        if d1c in ('flat', 'small_dip'):
            return {
                'action': 'ENTER_NOW',
                'grade': 'A',
                'detail': f'D1={d1:+.1f}% ({d1c}). {pred.get("note","")}. '
                          'BEST ENTRY POINT — buy at or near T+1 close.',
                'entry_timing': 'T+1 close — BUY NOW',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': pred.get('note', ''),
            }
        elif d1c == 'deep_dip':
            return {
                'action': 'ENTER_DIP',
                'grade': 'A-',
                'detail': f'D1={d1:+.1f}% (deep dip). {pred.get("note","")}. '
                          'Buy at T+1 close — D2 ALWAYS recovers in data.',
                'entry_timing': 'T+1 close — dip buy',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': pred.get('note', ''),
            }
        elif d1c == 'small_up':
            return {
                'action': 'ENTER_CAUTIOUS',
                'grade': 'B+',
                'detail': f'D1={d1:+.1f}% (small up). D2 positive 78% of time. '
                          'Enter at T+1 close or wait for T+2 open.',
                'entry_timing': 'T+1 close  OR  T+2 open',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': pred.get('note', ''),
            }
        else:  # strong_up
            return {
                'action': 'ENTER_CAUTIOUS',
                'grade': 'B-',
                'detail': f'D1={d1:+.1f}% (strong up — early movers beware). '
                          'D2 is 50/50. Do NOT chase. Wait for T+2 dip to enter.',
                'entry_timing': 'Wait for T+2 dip',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': pred.get('note', ''),
            }

    # ── T=2: D1 and D2 known ──────────────────────────────────────────────────
    if avail_days == 2 and d1 is not None and d2 is not None:
        s2 = encode_day(d1) + encode_day(d2)
        if s2 == 'UU':
            return {
                'action': 'HOLD',
                'grade': 'A+',
                'detail': 'Shape UU — UUUU/UUUF/UUUD all hit 100%. '
                          'Hold existing. If not in: enter at T+2 close.',
                'entry_timing': 'T+2 close (if not already in)',
                'stop_loss': stop,
                'hold_signal': True,
                'd1_prediction': 'D3 very likely positive',
            }
        elif s2 in ('FU', 'DU'):
            return {
                'action': 'HOLD_ENTER',
                'grade': 'A',
                'detail': f'Shape {s2} — strong recovery pattern. '
                          'FUUU=80%, FUUF=100%, DUUD=67%. Enter/hold at T+2 close.',
                'entry_timing': 'T+2 close',
                'stop_loss': stop,
                'hold_signal': True,
                'd1_prediction': 'D3 likely positive',
            }
        elif s2 == 'UD':
            return {
                'action': 'ENTER_DIP',
                'grade': 'B+',
                'detail': 'Shape UD — dip after rip. UDUU hits 100% avg +28%. '
                          'Buy the T+2 dip. Exit if D3 is also negative.',
                'entry_timing': 'T+2 low (dip buy)',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'D3 should recover — if not, exit',
            }
        elif s2 in ('FF', 'UF'):
            return {
                'action': 'ENTER',
                'grade': 'B+',
                'detail': f'Shape {s2} — slow build, D3 typically strong. '
                          'Enter at T+2 close. Expect D3/D4 to carry.',
                'entry_timing': 'T+2 close',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'D3 usually the acceleration day',
            }
        elif s2 in ('FD', 'DD', 'DF'):
            return {
                'action': 'SKIP',
                'grade': 'F',
                'detail': f'Shape {s2} — double negative / fading. '
                          'Pattern failing. DO NOT ENTER.',
                'entry_timing': 'DO NOT ENTER',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'High risk of continued decline',
            }

    # ── T=3: D1-D3 known ──────────────────────────────────────────────────────
    if avail_days == 3 and d3 is not None:
        s3 = encode_day(d1) + encode_day(d2) + encode_day(d3)
        stat = SHAPE_STATS.get(s3 + '?', {})
        if d3 > 2.0:
            return {
                'action': 'HOLD',
                'grade': 'A',
                'detail': f'Shape {s3}? — D3 strong (+{d3:.1f}%). '
                          'D4 very likely positive. Hold to T+4 at minimum.',
                'entry_timing': 'Hold to T+4',
                'stop_loss': stop,
                'hold_signal': True,
                'd1_prediction': f'D4 likely positive (D3 was strong)',
            }
        elif d3 < -2.0:
            return {
                'action': 'EXIT',
                'grade': 'D',
                'detail': f'Shape {s3}? — D3 reversal ({d3:+.1f}%). '
                          'Pattern breaking down. Exit or set tight stop.',
                'entry_timing': 'EXIT / tight stop',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'D4 uncertain — pattern compromised',
            }
        else:
            return {
                'action': 'HOLD_MONITOR',
                'grade': 'B',
                'detail': f'Shape {s3}? — D3 flat ({d3:+.1f}%). '
                          'Hold. Watch D4 for direction.',
                'entry_timing': 'Hold, watch D4',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'D4 will confirm trend',
            }

    # ── T=4: all 4 days known ─────────────────────────────────────────────────
    if avail_days >= 4 and d4 is not None:
        s4 = shape_fingerprint(d1, d2, d3, d4)
        stat = SHAPE_STATS.get(s4, {})
        hold_ext = d4 is not None and d3 is not None and d4 > 0 and d4 > d3
        still_up = d4 is not None and d4 > 2.0
        if hold_ext:
            return {
                'action': 'HOLD_EXTENDED',
                'grade': 'A+',
                'detail': f'Shape {s4} — ACCELERATING into T+4 (D4={d4:+.1f}>D3={d3:+.1f}). '
                          'T+5/T+6 very likely positive. HOLD BEYOND T+4.',
                'entry_timing': 'HOLD — extend to T+5/T+6',
                'stop_loss': stop,
                'hold_signal': True,
                'd1_prediction': f'T+5 likely positive — stock still building momentum',
            }
        elif still_up:
            return {
                'action': 'HOLD_T5',
                'grade': 'A',
                'detail': f'Shape {s4} — D4 still positive (+{d4:.1f}%). '
                          'Consider holding T+5.',
                'entry_timing': 'Consider T+5 hold',
                'stop_loss': stop,
                'hold_signal': True,
                'd1_prediction': 'T+5 possible continuation',
            }
        else:
            return {
                'action': 'EXIT',
                'grade': 'B',
                'detail': f'Shape {s4} — {stat.get("note","T+4 complete")}. '
                          'D4 fading. Plan exit at T+4 close.',
                'entry_timing': 'Exit at T+4 close',
                'stop_loss': stop,
                'hold_signal': False,
                'd1_prediction': 'T+5 risky — momentum fading',
            }

    return {
        'action': 'WATCH',
        'grade': 'B',
        'detail': 'Monitoring pattern.',
        'entry_timing': 'TBD',
        'stop_loss': stop,
        'hold_signal': False,
    }


# ── DATA LOADING ───────────────────────────────────────────────────────────────
def load_all():
    if not DATA.exists():
        sys.exit(f"ERROR: data directory not found: {DATA}")
    csv_files = sorted(DATA.glob("*/*/*.csv"))
    if not csv_files:
        sys.exit("ERROR: no CSV files found under data/equity/")
    print(f"  Found {len(csv_files)} CSV files (direct scan)")
    frames = []; n_skip = 0
    for p in csv_files:
        ds = p.stem
        if len(ds) != 10 or ds[4] != '-' or ds[7] != '-':
            n_skip += 1; continue
        try:
            df = pd.read_csv(p, low_memory=False)
            df.columns = df.columns.str.strip()
            cm = {}
            for col in df.columns:
                u = col.strip().upper()
                if u in ("SYMBOL","TCKRSYMB"):                cm[col]="sym"
                elif u in ("SERIES","SCTYSRS"):               cm[col]="series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):    cm[col]="o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):    cm[col]="h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):       cm[col]="l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"):  cm[col]="c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):        cm[col]="v"
            df = df.rename(columns=cm)
            if not {"sym","series","o","h","l","c","v"}.issubset(df.columns):
                n_skip += 1; continue
            df = df[df["series"].str.strip().isin(["EQ","BE"])].copy()
            df["date"] = pd.to_datetime(ds)
            frames.append(df[["sym","o","h","l","c","v","date"]])
        except:
            n_skip += 1
    if not frames: sys.exit("ERROR: no usable data found")
    if n_skip: print(f"  Skipped {n_skip} files")
    all_data = pd.concat(frames, ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")
    result = (all_data.dropna(subset=["o","h","l","c","v"])
              .sort_values(["sym","date"]).reset_index(drop=True))
    dates_found = sorted(result["date"].dt.strftime("%Y-%m-%d").unique())
    print(f"  Date range: {dates_found[0]}  →  {dates_found[-1]}  ({len(dates_found)} trading days)")
    print(f"  Latest 5 dates: {dates_found[-5:]}")
    return result


# ── CONSOLIDATION ──────────────────────────────────────────────────────────────
def find_consolidation(c_arr, start, length):
    window = c_arr[start: start + length]
    if len(window) < length: return None
    lo = float(np.min(window)); hi = float(np.max(window))
    if lo <= 0 or hi <= 0: return None
    rng = (hi - lo) / lo * 100
    return r2(lo), r2(hi), r2(rng)


# ── STOCK ANALYSIS ─────────────────────────────────────────────────────────────
def analyze_stock(sym, df, latest_set, all_dates):
    n     = len(df)
    o_arr = df["o"].values; h_arr = df["h"].values
    l_arr = df["l"].values; c_arr = df["c"].values; v_arr = df["v"].values
    dates = pd.to_datetime(df["date"].values)

    cur_price = float(c_arr[-1])
    last_date = str(dates[-1].date())

    if cur_price < MIN_PRICE:                                    return None
    if last_date not in latest_set:                              return None
    if len(set(int(d.year) for d in dates)) < MIN_YEARS:        return None

    tv5 = [float(c_arr[j]) * float(v_arr[j])
           for j in range(max(0, n-5), n) if float(v_arr[j]) > 0]
    if not tv5 or sum(tv5)/len(tv5) < MIN_TURNOVER:             return None
    turnover_cr = r2(sum(tv5)/len(tv5)/1e7)

    breakouts = []
    for cw in CONSOL_WINDOWS:
        start = 0
        while start < n - cw:
            brk = start + cw
            res = find_consolidation(c_arr, start, cw)
            if res is None:          start += 1; continue
            consol_lo, consol_hi, consol_rng = res
            if consol_rng > CONSOL_BAND_MAX or consol_rng <= 0: start += 1; continue
            if brk >= n:             start += 1; continue

            brk_close = float(c_arr[brk])
            brk_pct   = (brk_close - consol_hi) / consol_hi * 100
            if brk_pct < BREAKOUT_MIN: start += 1; continue

            vol_win   = v_arr[max(0, brk-20): brk]
            vol_ma    = float(np.mean(vol_win)) if len(vol_win) > 0 else float(v_arr[brk])
            vol_ratio = float(v_arr[brk]) / vol_ma if vol_ma > 0 else 1.0

            post_rets = {}
            for d in range(1, POST_DAYS + 1):
                xi = brk + d
                if xi >= n: break
                post_rets[d] = r2((float(c_arr[xi]) - brk_close) / brk_close * 100)

            t1 = brk + 1
            t1_open_gap = r2((float(o_arr[t1]) - brk_close) / brk_close * 100) if t1 < n else None
            t1_dip_pct  = r2((float(l_arr[t1]) - brk_close) / brk_close * 100) if t1 < n else None

            avail_days = min(POST_DAYS, n - brk - 1)
            pos_days   = sum(1 for d in range(1, avail_days+1)
                            if (post_rets.get(d) or -999) > 0)
            if avail_days >= MIN_POST_POSITIVE and pos_days < MIN_POST_POSITIVE:
                start += 1; continue

            # Daily moves and shape
            d1, d2, d3, d4 = daily_moves(post_rets)
            shape = shape_fingerprint(d1, d2, d3, d4)

            avail_vals = [v for v in [post_rets.get(i) for i in range(1,5)] if v is not None]
            peak_ret   = r2(max(avail_vals)) if avail_vals else None
            # T+5/6 hold signal
            hold_sig   = (d3 is not None and d4 is not None and d3 > 0 and d4 > 0)
            accel_sig  = (d3 is not None and d4 is not None and d4 > d3 > 0)

            breakouts.append({
                "date":        str(dates[brk].date()),
                "year":        int(dates[brk].year),
                "consol_days": cw,
                "consol_lo":   consol_lo,
                "consol_hi":   consol_hi,
                "consol_rng":  consol_rng,
                "brk_close":   r2(brk_close),
                "brk_pct":     r2(brk_pct),
                "vol_ratio":   r2(vol_ratio),
                "t1_open_gap": t1_open_gap,
                "t1_dip_pct":  t1_dip_pct,
                "post_rets":   post_rets,
                "d1":d1,"d2":d2,"d3":d3,"d4":d4,
                "shape":       shape,
                "peak_ret":    peak_ret,
                "avail_days":  avail_days,
                "hold_signal": hold_sig,
                "accel_signal":accel_sig,
                "timing_cat":  timing_category(r2(brk_pct)),
            })
            start += max(cw // 2, 5)

    if len(breakouts) < MIN_OCCURRENCES: return None

    # Deduplicate
    breakouts.sort(key=lambda x: x["date"])
    deduped = [breakouts[0]]
    for b in breakouts[1:]:
        try:
            prev_d = datetime.strptime(deduped[-1]["date"], "%Y-%m-%d")
            this_d = datetime.strptime(b["date"],           "%Y-%m-%d")
            if (this_d - prev_d).days < 5:
                if b["brk_pct"] > deduped[-1]["brk_pct"]: deduped[-1] = b
            else:
                deduped.append(b)
        except: deduped.append(b)

    if len(deduped) < MIN_OCCURRENCES: return None

    # Quality filter — 75% of historical breakouts must peak >= 5%
    def peak_of(b):
        pr   = b.get("post_rets", {})
        vals = [pr[d] for d in range(1, POST_DAYS+1) if pr.get(d) is not None]
        return max(vals) if vals else None

    historical = [b for b in deduped if b.get("avail_days",0) >= POST_DAYS]
    min_peak_ret = avg_peak_ret = None
    if historical:
        peaks     = [p for b in historical if (p := peak_of(b)) is not None]
        if peaks:
            qualifying = sum(1 for p in peaks if p >= MIN_GUARANTEED_PEAK)
            if qualifying / len(peaks) < PEAK_QUALIFY_RATE: return None
            min_peak_ret = r2(min(peaks))
            avg_peak_ret = r2(sum(peaks)/len(peaks))

    for b in deduped:
        b["peak_ret"] = r2(peak_of(b))

    # Aggregate stats
    def safe_avg(vals):
        v = [x for x in vals if x is not None]
        return r2(sum(v)/len(v)) if v else None

    avg_brk_pct    = safe_avg([b["brk_pct"]        for b in deduped])
    avg_t1_dip     = safe_avg([b["t1_dip_pct"]     for b in deduped])
    avg_t1_gap     = safe_avg([b["t1_open_gap"]    for b in deduped])
    avg_consol_rng = safe_avg([b["consol_rng"]     for b in deduped])
    avg_consol_days= safe_avg([b["consol_days"]    for b in deduped])
    avg_t1_ret     = safe_avg([b["post_rets"].get(1) for b in deduped])
    avg_t4_ret     = safe_avg([b["post_rets"].get(4) for b in deduped])
    avg_d1         = safe_avg([b["d1"] for b in deduped])
    avg_d2         = safe_avg([b["d2"] for b in deduped])
    avg_d3         = safe_avg([b["d3"] for b in deduped])
    avg_d4         = safe_avg([b["d4"] for b in deduped])

    # Historical shape distribution for this stock
    shape_dist = {}
    for b in historical:
        s = b.get("shape","????")
        if '?' not in s:
            shape_dist[s] = shape_dist.get(s, 0) + 1

    if avg_t1_dip is not None and avg_t1_dip < -1.0:
        entry_rec  = f"Wait for T+1 dip to ~{avg_t1_dip:.1f}% below signal close"
        entry_type = "wait_for_dip"
    elif avg_t1_gap is not None and avg_t1_gap < 0:
        entry_rec  = f"Enter at T+1 open (opens {avg_t1_gap:+.1f}% below signal close)"
        entry_type = "t1_open"
    else:
        entry_rec  = "Enter at T+1 open (opens near/above signal close)"
        entry_type = "t1_open"

    # Alerts
    recent_set = set(all_dates[-ALERT_WINDOW:]) if len(all_dates) >= ALERT_WINDOW else set(all_dates)
    alerts = []
    for b in reversed(deduped):
        if b["date"] not in recent_set: continue
        try:
            days_since = (now_ist.replace(tzinfo=None) -
                          datetime.strptime(b["date"], "%Y-%m-%d")).days
        except: days_since = 0

        sig_idx = next((j for j, d in enumerate(dates)
                        if str(d.date()) == b["date"]), None)
        t1_date = str(dates[sig_idx+1].date()) if sig_idx is not None and sig_idx+1 < n else "tomorrow"
        t2_date = str(dates[sig_idx+2].date()) if sig_idx is not None and sig_idx+2 < n else "day after"

        # Compute how many days of post-data we have for THIS alert
        alert_avail = b.get("avail_days", 0)

        # Decision system
        dec = entry_decision(b["d1"], b["d2"], b["d3"], b["d4"],
                             b["brk_pct"], alert_avail, b["brk_close"])

        if days_since == 0:   action = "TODAY — do not enter yet, watch D1"; buy_on = "WATCH"
        elif days_since <= 2: action = f"Signal {days_since}d ago — see entry decision"; buy_on = "ACTING"
        else:                 action = f"Signal {days_since}d ago — late stage"; buy_on = "LATE"

        # Partial shape (only known days)
        known_days = [b["d1"], b["d2"], b["d3"], b["d4"]]
        partial_shape = ''.join(encode_day(v) if v is not None else '.' for v in known_days)

        # Shape grade from known 4-day shape (if complete)
        shape_stat = SHAPE_STATS.get(b.get("shape","????"), {})

        alerts.append({
            "sig_date":    b["date"],
            "days_since":  days_since,
            "brk_close":   b["brk_close"],
            "brk_pct":     b["brk_pct"],
            "consol_days": b["consol_days"],
            "consol_rng":  b["consol_rng"],
            "vol_ratio":   b["vol_ratio"],
            "t1_date":     t1_date,
            "t2_date":     t2_date,
            "t1_open_gap": b["t1_open_gap"],
            "t1_dip_pct":  b["t1_dip_pct"],
            "post_rets":   b["post_rets"],
            "d1":b["d1"],"d2":b["d2"],"d3":b["d3"],"d4":b["d4"],
            "partial_shape": partial_shape,
            "shape":       b.get("shape","????"),
            "shape_grade": shape_stat.get("grade","?"),
            "shape_note":  shape_stat.get("note",""),
            "timing_cat":  b.get("timing_cat","standard"),
            "peak_ret":    b.get("peak_ret"),
            "hold_signal": b.get("hold_signal", False),
            "accel_signal":b.get("accel_signal", False),
            "action":      action,
            "buy_on":      buy_on,
            "entry_type":  entry_type,
            "entry_rec":   entry_rec,
            "avg_t4_ret":  avg_t4_ret,
            "min_peak_ret":min_peak_ret,
            "decision":    dec,
        })

    return {
        "sym":             sym,
        "price":           r2(cur_price),
        "last_date":       last_date,
        "turnover_cr":     turnover_cr,
        "n_breakouts":     len(deduped),
        "years":           sorted(set(b["year"] for b in deduped)),
        "avg_brk_pct":     avg_brk_pct,
        "avg_consol_rng":  avg_consol_rng,
        "avg_consol_days": avg_consol_days,
        "avg_t1_dip":      avg_t1_dip,
        "avg_t1_gap":      avg_t1_gap,
        "avg_t1_ret":      avg_t1_ret,
        "avg_t4_ret":      avg_t4_ret,
        "avg_d1":avg_d1,"avg_d2":avg_d2,"avg_d3":avg_d3,"avg_d4":avg_d4,
        "min_peak_ret":    min_peak_ret,
        "avg_peak_ret":    avg_peak_ret,
        "timing_cat":      timing_category(avg_brk_pct),
        "shape_dist":      shape_dist,
        "entry_type":      entry_type,
        "entry_rec":       entry_rec,
        "has_alert":       len(alerts) > 0,
        "alerts":          alerts,
        "breakouts":       sorted(deduped, key=lambda x: x["date"], reverse=True),
    }


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}\nNSE Breakout Analyzer  |  {now_str} IST")
    print(f"Quality: {PEAK_QUALIFY_RATE*100:.0f}% of historical breakouts must peak >= {MIN_GUARANTEED_PEAK}%")
    print(f"Alert window: last {ALERT_WINDOW} trading days\n{'='*60}")

    all_data   = load_all()
    grouped    = all_data.groupby("sym")
    syms       = sorted(grouped.groups.keys())
    all_dates  = sorted(all_data["date"].dt.strftime("%Y-%m-%d").unique())
    latest_set = set(all_dates[-RECENT_DAYS:])
    last_fetch = all_dates[-1]

    print(f"Symbols: {len(syms):,}  |  Latest: {last_fetch}")
    print(f"Alert window: {all_dates[-ALERT_WINDOW]}  →  {last_fetch}\n")

    results = []; skipped = excluded = 0
    for i, sym in enumerate(syms):
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(syms)} — {len(results)} qualified")
        sym_up = sym.upper()
        if any(ex in sym_up for ex in EXCL_CONTAINS):
            excluded += 1; continue
        try:
            grp = (grouped.get_group(sym).sort_values("date").reset_index(drop=True))
            if len(grp) < 150: skipped += 1; continue
            res = analyze_stock(sym, grp, latest_set, all_dates)
            if res: results.append(res)
            else:   skipped += 1
        except: skipped += 1

    alert_stocks = sorted([r for r in results if r["has_alert"]],
                          key=lambda x: -(x.get("turnover_cr") or 0))
    all_stocks   = sorted(results,
                          key=lambda x: -(x.get("min_peak_ret") or x.get("avg_t4_ret") or 0))

    all_alerts = []
    for r in results:
        for a in (r.get("alerts") or []):
            all_alerts.append({
                **a, "sym":r["sym"], "price":r["price"],
                "turnover_cr":r["turnover_cr"],
                "avg_t1_dip":r["avg_t1_dip"],
                "avg_peak_ret":r["avg_peak_ret"],
                "entry_rec":r["entry_rec"],
                "entry_type":r["entry_type"],
            })
    all_alerts.sort(key=lambda x: (x.get("days_since") or 99, -(x.get("turnover_cr") or 0)))

    output = {
        "generated_at":now_str, "today_ist":today, "last_fetch":last_fetch,
        "alert_window": f"{all_dates[-ALERT_WINDOW]} to {last_fetch}",
        "n_stocks":len(results), "n_alerts":len(alert_stocks),
        "n_all_alerts":len(all_alerts),
        "alert_stocks":alert_stocks, "all_alerts":all_alerts, "stocks":all_stocks,
        "shape_stats": SHAPE_STATS,
        "d1_predictions": D1_D2_PREDICTION,
    }

    (OUT / "breakout_signals.json").write_text(json.dumps(output, indent=2))
    print(f"\n  Qualified: {len(results)}  |  Alerts: {len(alert_stocks)}  |  "
          f"Total signals: {len(all_alerts)}  |  Skipped: {skipped}")
    if alert_stocks:
        print(f"\n{'='*60}\nTOP ALERTS")
        for r in alert_stocks[:15]:
            a  = r["alerts"][0]
            dec = a.get("decision", {})
            print(f"  {r['sym']:<14} brk={a['brk_pct']:+.1f}%  "
                  f"shape={a.get('partial_shape','????')}  "
                  f"grade={dec.get('grade','?')}  "
                  f"→ {dec.get('action','?')}  {a['sig_date']}")

if __name__ == "__main__":
    main()
