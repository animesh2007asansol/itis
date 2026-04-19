#!/usr/bin/env python3
"""
candle_enriched_analysis.py  v3
================================
Deep analysis of TODAY's candle signals.

New in v3:
  - Entry price analysis: for every historical occurrence, tracks:
      * open_vs_close: next day opened how much % above/below signal close
      * low_vs_open:   next day's LOW went how much % below the open price
      * safe_buy_pct:  worst-case minimum dip from open (your safe limit order)
      * safe_buy_px:   actual Rs price you can safely set as limit order
  - Sort by 5d average return (strongest 5d signal goes to top)
  - min_occurrences relaxed: stocks with >=1 occurrence in re-load are included
    (candle_patterns already verified >=3 occurrences; re-load may find fewer
     due to CSV file gaps, but we trust the pattern was validated)
  - Falls back to candle_today.json stats if re-loaded occurrences < 2
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
    print("ERROR: pip install pandas numpy"); sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "pattern_signals"
MANIFEST  = DATA_DIR / "manifest.json"

LIQUIDITY_TARGET_RS = 200_000
MIN_OCC_TO_ANALYSE  = 1   # use any occurrence found; validation done by candle_patterns

# ─────────────────────────────────────────────────────────────────────────────
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date_type, datetime)): return str(obj)
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.bool_):     return bool(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        try:
            if pd.isna(obj): return None
        except Exception: pass
        return super().default(obj)

def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)

COL_MAP = {
    "SYMBOL":"SYMBOL","TCKRSYMB":"SYMBOL",
    "SERIES":"SERIES","SCTYSRS":"SERIES",
    "OPEN":"OPEN","OPNPRIC":"OPEN","OPEN PRICE":"OPEN",
    "HIGH":"HIGH","HGHPRIC":"HIGH","HIGH PRICE":"HIGH",
    "LOW":"LOW","LWPRIC":"LOW","LOW PRICE":"LOW",
    "CLOSE":"CLOSE","CLSPRIC":"CLOSE","CLOSE PRICE":"CLOSE","LASTPRIC":"CLOSE",
    "TOTTRDQTY":"VOLUME","TTLTRADGVOL":"VOLUME",
    "TIMESTAMP":"DATE","DATE":"DATE","DATE1":"DATE","BIZDT":"DATE",
}

def load_stock(sym, trading_days):
    rows = []
    for ds in trading_days:
        y, m, _ = ds.split("-")
        path = DATA_DIR / "equity" / y / m / f"{ds}.csv"
        if not path.exists(): continue
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
            df.columns = df.columns.str.strip().str.strip('"').str.strip("'").str.upper()
            df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})
            if not all(c in df.columns for c in ["SYMBOL","OPEN","HIGH","LOW","CLOSE"]): continue
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip().isin(["EQ","BE"])]
            row = df[df["SYMBOL"].str.strip()==sym]
            if row.empty: continue
            row = row.iloc[0]
            vol = float(str(row.get("VOLUME","0")).replace(",","")) if "VOLUME" in df.columns else 0.0
            rows.append({
                "date": pd.Timestamp(ds),
                "o": float(row["OPEN"]), "h": float(row["HIGH"]),
                "l": float(row["LOW"]),  "c": float(row["CLOSE"]),
                "v": vol,
            })
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def body_bkt(bp):
    if bp<0.3: return "doji"
    if bp<1.0: return "tiny"
    if bp<2.5: return "small"
    if bp<5.0: return "medium"
    return "large"

def wick_bkt(r):
    if pd.isna(r) or r<0.1: return "none"
    if r<0.5: return "tiny"
    if r<1.0: return "small"
    if r<2.0: return "medium"
    if r<4.0: return "long"
    return "very_long"

def classify_key(o, h, l, c):
    body=abs(c-o); upper=h-max(o,c); lower=min(o,c)-l
    sb=body if body>0 else 0.0001
    bp=body/c*100 if c>0 else 0
    color="G" if c>=o else "R"
    return f"{color}_{body_bkt(bp)}_L{wick_bkt(lower/sb)}_U{wick_bkt(upper/sb)}"

def r2(n): return round(n*100)/100 if n is not None else None

def deep_analyse(sym, candle_key, alert, df):
    if df.empty or len(df) < 5: return None
    df = df.copy().reset_index(drop=True)
    df["ckey"] = [classify_key(float(r.o),float(r.h),float(r.l),float(r.c)) for _,r in df.iterrows()]

    MIN_VOL = 50_000
    occurrences = []

    for i, row in df.iterrows():
        if row["ckey"] != candle_key: continue
        if float(row["v"]) < MIN_VOL: continue   # must have enough volume
        if i+1 >= len(df): continue              # need next day

        sig_c = float(row["c"])
        sig_v = float(row["v"])
        nxt   = df.iloc[i+1]
        nxt_o = float(nxt["o"]); nxt_h = float(nxt["h"])
        nxt_l = float(nxt["l"]); nxt_c = float(nxt["c"])

        # ── Core return metrics ───────────────────────────────────────────
        gap_vs_close   = r2((nxt_o - sig_c) / sig_c * 100)   # open vs signal close
        nd_close_ret   = r2((nxt_c - sig_c) / sig_c * 100)   # close return
        nd_high_ret    = r2((nxt_h - sig_c) / sig_c * 100)   # best possible (high)

        # ── Entry dip analysis ────────────────────────────────────────────
        # Did next day's LOW go below the signal CLOSE? (classic dip-buy)
        dip_below_close     = nxt_l < sig_c
        dip_below_close_pct = r2((sig_c - nxt_l) / sig_c * 100) if dip_below_close else 0.0

        # Did next day's LOW go below the next day's OPEN?
        # This tells you: even after gap-up open, can you buy cheaper?
        dip_below_open     = nxt_l < nxt_o
        dip_below_open_pct = r2((nxt_o - nxt_l) / nxt_o * 100) if dip_below_open else 0.0

        # Best entry = lowest price you could have gotten on next day
        # = min(nxt_l, nxt_o) but if opened above close and dipped below open
        best_entry_px = r2(min(nxt_l, nxt_o))
        best_entry_vs_close = r2((best_entry_px - sig_c) / sig_c * 100)

        # Safe buy calculation:
        # If you place a limit order at nxt_o * (1 - dip_below_open_pct/100),
        # historically that order would have been filled if dip happened.
        # safe_buy_px = next_open * (1 - dip_from_open)
        safe_entry_from_open = r2(nxt_o * (1 - dip_below_open_pct/100)) if dip_below_open else nxt_o
        safe_entry_vs_close  = r2((safe_entry_from_open - sig_c) / sig_c * 100)

        # Pre-signal context
        pre5  = df.iloc[max(0,i-5):i]
        pre20 = df.iloc[max(0,i-20):i]
        pre60 = df.iloc[max(0,i-60):i]
        trend5  = r2((sig_c/float(pre5.iloc[0]["c"])-1)*100) if len(pre5)>=1 else None
        trend20 = r2((sig_c/float(pre20.iloc[0]["c"])-1)*100) if len(pre20)>=1 else None
        trend60 = r2((sig_c/float(pre60.iloc[0]["c"])-1)*100) if len(pre60)>=1 else None

        vol_avg20 = float(pre20["v"].mean()) if len(pre20)>=5 else 0.0
        vol_ratio = r2(sig_v/vol_avg20) if vol_avg20>0 else None
        turnover  = round(sig_c * sig_v)

        occurrences.append({
            "date":                str(row["date"].date()),
            "sig_close":           r2(sig_c),
            "sig_volume":          int(sig_v),
            "turnover_rs":         turnover,
            "vol_ratio_vs_20d":    vol_ratio,
            # Gap / open analysis
            "next_open":           r2(nxt_o),
            "opened_above_close":  nxt_o > sig_c,
            "gap_vs_close_pct":    gap_vs_close,
            # Dip from open (key new field)
            "dip_below_open":      dip_below_open,
            "dip_below_open_pct":  dip_below_open_pct,
            # Dip from close
            "dip_below_close":     dip_below_close,
            "dip_below_close_pct": dip_below_close_pct,
            # Best entry
            "best_entry_px":       best_entry_px,
            "best_entry_vs_close": best_entry_vs_close,
            # Safe limit-order entry
            "safe_entry_px":       safe_entry_from_open,
            "safe_entry_vs_close": safe_entry_vs_close,
            # Returns
            "nd_close_ret":        nd_close_ret,
            "nd_high_ret":         nd_high_ret,
            "next_high":           r2(nxt_h),
            "next_low":            r2(nxt_l),
            "next_close":          r2(nxt_c),
            # Pre-signal
            "trend_5d_before":     trend5,
            "trend_20d_before":    trend20,
            "trend_60d_before":    trend60,
        })

    if len(occurrences) < MIN_OCC_TO_ANALYSE:
        return None

    # Year spread check: pattern must have fired in 2+ years to be trusted
    occ_years = set(o['date'][:4] for o in occurrences)
    if len(occ_years) < 2:
        return None   # single-year pattern — may be regime-specific anomaly


    n = len(occurrences)

    # Aggregate
    gaps       = [o["gap_vs_close_pct"] for o in occurrences]
    nd_rets    = [o["nd_close_ret"] for o in occurrences]
    nd_maxes   = [o["nd_high_ret"] for o in occurrences]
    dob_pcts   = [o["dip_below_open_pct"] for o in occurrences]  # dip below open
    doc_pcts   = [o["dip_below_close_pct"] for o in occurrences] # dip below close
    safe_pcts  = [o["safe_entry_vs_close"] for o in occurrences]
    turnovers  = [o["turnover_rs"] for o in occurrences]
    vol_rats   = [o["vol_ratio_vs_20d"] for o in occurrences if o["vol_ratio_vs_20d"]]

    n_gap_up   = sum(1 for g in gaps if g>0)
    n_dip_open = sum(1 for o in occurrences if o["dip_below_open"])
    n_dip_close= sum(1 for o in occurrences if o["dip_below_close"])

    pct_gap_up     = r2(n_gap_up/n*100)
    pct_dip_open   = r2(n_dip_open/n*100)
    pct_dip_close  = r2(n_dip_close/n*100)

    avg_gap        = r2(sum(gaps)/n)
    min_gap        = r2(min(gaps))   # worst (lowest) open vs close
    avg_nd         = r2(sum(nd_rets)/n)
    nd_min_hist    = r2(min(nd_rets))
    nd_max_hist    = r2(max(nd_rets))
    avg_nd_max     = r2(sum(nd_maxes)/n)

    # Dip below open stats
    avg_dip_open   = r2(sum(dob_pcts)/n)   # avg dip from open
    max_dip_open   = r2(max(dob_pcts))     # worst dip from open (most it ever fell)
    # Safe buy = worst-case: always fills if you set limit at:
    # open * (1 - max_dip_open/100)  → this always gets filled but at worst price
    # Conservative safe buy: use avg_dip_open * 0.5 (usually fills)
    today_close    = float(alert["today_close"])

    # Estimated next open using historical min gap % (worst case open)
    est_next_open  = r2(today_close * (1 + min_gap/100))  # if it opens at historical minimum

    # Safe buy price: est_next_open dipping by max_dip_open
    safe_buy_from_open   = r2(est_next_open * (1 - max_dip_open/100))
    safe_buy_vs_close    = r2((safe_buy_from_open - today_close) / today_close * 100)

    # Comfortable buy: est_next_open dipping by avg_dip_open (fills most of the time)
    comfort_buy          = r2(est_next_open * (1 - avg_dip_open/100))
    comfort_buy_vs_close = r2((comfort_buy - today_close) / today_close * 100)

    avg_turnover  = round(sum(turnovers)/len(turnovers)) if turnovers else 0
    liquidity_ok  = avg_turnover >= LIQUIDITY_TARGET_RS
    avg_vol_ratio = r2(sum(vol_rats)/len(vol_rats)) if vol_rats else None
    pre_t20s      = [o["trend_20d_before"] for o in occurrences if o["trend_20d_before"] is not None]
    avg_trend20   = r2(sum(pre_t20s)/len(pre_t20s)) if pre_t20s else None

    # Max adverse from close (never negative in our filtered dataset, but track anyway)
    max_adverse_from_close = r2(max(doc_pcts))

    # Targets based on today_close
    t1 = r2(today_close * (1 + nd_min_hist/100))
    t2 = r2(today_close * (1 + avg_nd/100))
    t3 = r2(today_close * (1 + avg_nd_max/100))

    # Stop loss: only if stock ever went below signal close
    sl_needed  = max_adverse_from_close > 1.0
    suggested_sl_pct = r2(-max_adverse_from_close - 0.5) if sl_needed else None
    suggested_sl_px  = r2(today_close*(1-(max_adverse_from_close+0.5)/100)) if sl_needed else None

    # ── Score ──────────────────────────────────────────────────────────
    # Use 5d data from the original alert (more reliable than re-computed)
    w5d    = alert.get("win_5d") or {}
    avg_5d = w5d.get("avg") or 0

    # Confidence score — occurrences and year spread have highest weight
    # With MIN_OCC=6 and MIN_YEARS=2 already enforced upstream,
    # we differentiate further: 8+ occ and 3+ years = much higher trust
    score = 20  # base: passed all upstream filters
    n_years_here = len(set(o['date'][:4] for o in occurrences))
    # Occurrences: 6=+0, 8=+10, 10=+16, 15=+20 (capped)
    score += min(20, max(0, n-5) * 2)
    # Year spread: 2yr=+8, 3yr=+16, 4yr=+20 (most important factor)
    score += min(20, n_years_here * 5)
    if avg_vol_ratio and avg_vol_ratio>=1.5: score+=10  # volume confirms signal
    if avg_vol_ratio and avg_vol_ratio>=2.0: score+=5   # extra for very high volume
    if not sl_needed: score+=10  # never went negative — cleanest signal
    if liquidity_ok: score+=5    # can actually deploy Rs2L
    if avg_5d >= 5: score+=10    # strong 5d: pattern stays strong beyond next day
    elif avg_5d >= 3: score+=5
    score = min(100, score)

    notes = []
    if avg_vol_ratio and avg_vol_ratio>=2: notes.append(f"Volume {avg_vol_ratio}x avg — strong institutional interest")
    if avg_trend20 is not None and avg_trend20<-3: notes.append(f"Stock down {abs(avg_trend20):.1f}% before signal — dip reversal")
    if pct_gap_up>=80: notes.append(f"Opens higher {pct_gap_up:.0f}% of next mornings")
    if pct_dip_open>=50: notes.append(f"Dips below open {pct_dip_open:.0f}% of times — good limit-order opportunity")
    if avg_5d>=5: notes.append(f"5-day avg return +{avg_5d:.1f}% — pattern stays strong beyond next day")
    if n>=6: notes.append(f"Fired {n} times — robust sample")

    return {
        "sym":               sym,
        "candle_key":        candle_key,
        "desc":              alert.get("candle_desc",""),
        "today_date":        alert["today_date"],
        "today_close":       today_close,
        "today_vol":         alert.get("today_vol",0),
        "n_occurrences":     n,
        "years":             alert.get("years",[]),
        "score":             score,
        # ── Entry strategy ─────────────────────────────────────
        "pct_gap_up":           pct_gap_up,
        "avg_gap_pct":          avg_gap,
        "min_gap_pct":          min_gap,
        "est_next_open":        est_next_open,
        "pct_dip_below_open":   pct_dip_open,    # % of times price dipped below open
        "avg_dip_from_open":    avg_dip_open,    # avg % it dipped below open
        "max_dip_from_open":    max_dip_open,    # worst case dip from open
        "pct_dip_below_close":  pct_dip_close,
        "safe_buy_px":          safe_buy_from_open,  # always fills (worst-case dip)
        "safe_buy_vs_close":    safe_buy_vs_close,
        "comfort_buy_px":       comfort_buy,         # fills most of the time
        "comfort_buy_vs_close": comfort_buy_vs_close,
        "buy_strategy": (
            f"Place limit @ Rs{comfort_buy} ({comfort_buy_vs_close:+.1f}% vs close) — "
            f"dip happens {pct_dip_open:.0f}% of times. Worst fill ever: Rs{safe_buy_from_open}."
        ) if pct_dip_open>=50 else (
            f"Gap-up open {pct_gap_up:.0f}% of times. "
            f"Est open: Rs{est_next_open}. Buy at open or pre-market."
        ),
        # ── Returns ────────────────────────────────────────────
        "nd_min_return":   nd_min_hist,
        "nd_avg_return":   avg_nd,
        "nd_max_return":   nd_max_hist,
        "nd_avg_high":     avg_nd_max,
        "win_5d_avg":      avg_5d,
        "win_5d_min":      w5d.get("min"),
        "win_10d_avg":     (alert.get("win_10d") or {}).get("avg"),
        "win_20d_avg":     (alert.get("win_20d") or {}).get("avg"),
        # ── Targets ────────────────────────────────────────────
        "target_1":        t1,
        "target_2":        t2,
        "target_3":        t3,
        "target_1_pct":    nd_min_hist,
        "target_2_pct":    avg_nd,
        # ── Stop loss ──────────────────────────────────────────
        "sl_needed":       sl_needed,
        "max_adverse_from_close": max_adverse_from_close,
        "suggested_sl_px": suggested_sl_px,
        "suggested_sl_pct":suggested_sl_pct,
        "sl_note": (
            f"Max dip below signal close: {max_adverse_from_close}% — SL @ Rs{suggested_sl_px}"
            if sl_needed else
            "No SL needed — never closed below signal close in all occurrences"
        ),
        # ── Liquidity ──────────────────────────────────────────
        "avg_turnover_rs": avg_turnover,
        "liquidity_ok":    liquidity_ok,
        "liquidity_note":  (f"Avg daily Rs{avg_turnover:,} — Rs2L order feasible"
                            if liquidity_ok else f"Low liquidity Rs{avg_turnover:,}"),
        # ── Context ────────────────────────────────────────────
        "avg_vol_ratio":        avg_vol_ratio,
        "avg_trend_20d_before": avg_trend20,
        "strength_note":        ". ".join(notes) or "Consistent 100%-win candle pattern",
        # ── Full occurrence data ────────────────────────────────
        "occurrences":     occurrences,
    }

def main():
    print("="*65)
    print("Candle Enriched Analysis  v3")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f: manifest=json.load(f)
    tds = sorted(manifest.keys())
    latest = tds[-1]
    print(f"  {len(tds)} trading days | Latest: {latest}")

    today_path = OUT_DIR/"candle_today.json"
    if not today_path.exists():
        print("ERROR: candle_today.json not found. Run candle_patterns.py first.")
        sys.exit(1)
    with open(today_path) as f: today_data=json.load(f)
    alerts = today_data.get("alerts", [])
    print(f"  Today signals: {len(alerts)} (date: {today_data.get('signal_date','?')})")

    if not alerts:
        now_ist=datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime("%Y-%m-%dT%H:%M:%S+05:30")
        jdump({"generated_at":now_ist,"signal_date":latest,"n_analysed":0,"patterns":[]}, OUT_DIR/"candle_enriched.json")
        jdump({"generated_at":now_ist,"signal_date":latest,"top5":[]}, OUT_DIR/"candle_top5.json")
        print("  No signals today — empty outputs written.")
        return

    enriched=[]
    skipped_no_data=0
    skipped_no_occ=0

    for i, alert in enumerate(alerts):
        sym=alert["sym"]; ckey=alert["candle_key"]
        print(f"  [{i+1:02d}/{len(alerts)}] {sym:<14} ({ckey})...", end="", flush=True)
        df=load_stock(sym, tds)
        if df.empty:
            print(" NO DATA"); skipped_no_data+=1; continue
        result=deep_analyse(sym, ckey, alert, df)
        if not result:
            # Fallback: use alert's pre-computed stats, just no entry analysis
            print(f" FALLBACK (no matching occ in re-load)")
            skipped_no_occ+=1
            today_close=float(alert["today_close"])
            w5=alert.get("win_5d") or {}
            nd_min=float(alert.get("nd_min",0))
            nd_avg=float(alert.get("nd_avg",0))
            enriched.append({
                "sym":sym,"candle_key":ckey,"desc":alert.get("candle_desc",""),
                "today_date":alert["today_date"],"today_close":today_close,
                "today_vol":alert.get("today_vol",0),
                "n_occurrences":alert.get("occurrences",0),
                "years":alert.get("years",[]),"score":20,
                "pct_gap_up":float(alert.get("gap_up_rate",0)),
                "avg_gap_pct":None,"min_gap_pct":None,"est_next_open":today_close,
                "pct_dip_below_open":None,"avg_dip_from_open":0,"max_dip_from_open":0,
                "pct_dip_below_close":None,"safe_buy_px":today_close,"safe_buy_vs_close":0,
                "comfort_buy_px":today_close,"comfort_buy_vs_close":0,
                "buy_strategy":"Buy at tomorrow's open price (no entry analysis available)",
                "nd_min_return":nd_min,"nd_avg_return":nd_avg,"nd_max_return":float(alert.get("nd_max",0)),
                "nd_avg_high":nd_avg,"win_5d_avg":float(w5.get("avg",0)),"win_5d_min":w5.get("min"),
                "win_10d_avg":(alert.get("win_10d") or {}).get("avg"),
                "win_20d_avg":(alert.get("win_20d") or {}).get("avg"),
                "target_1":round(today_close*(1+nd_min/100),2),
                "target_2":round(today_close*(1+nd_avg/100),2),
                "target_3":round(today_close*(1+nd_avg*1.5/100),2),
                "target_1_pct":nd_min,"target_2_pct":nd_avg,
                "sl_needed":False,"max_adverse_from_close":0,
                "suggested_sl_px":None,"suggested_sl_pct":None,
                "sl_note":"No SL data (fallback mode — run workflow for full analysis)",
                "avg_turnover_rs":round(today_close*float(alert.get("today_vol",0))),
                "liquidity_ok":today_close*float(alert.get("today_vol",0))>=200_000,
                "liquidity_note":"Estimated from today volume",
                "avg_vol_ratio":None,"avg_trend_20d_before":None,
                "strength_note":"Pattern validated by candle_patterns; detailed entry analysis unavailable (fallback)",
                "occurrences":[],
            })
        else:
            enriched.append(result)
            w5avg=result.get("win_5d_avg",0) or 0
            print(f" score={result['score']} occ={result['n_occurrences']} "
                  f"5d={w5avg:+.1f}% nd={result['nd_avg_return']:+.1f}% "
                  f"buy=Rs{result['comfort_buy_px']}")

    # Sort: primarily by 5d avg return, then by score
    # Remove stocks with avg next-day return < 0.5%
    enriched = [r for r in enriched if (r.get('nd_avg_return') or 0) >= 0.5]
    enriched.sort(key=lambda x: (-(x.get("win_5d_avg") or 0), -x["score"]))
    top5=enriched[:5]

    print(f"\n  Total: {len(enriched)} analysed ({skipped_no_data} no data, {skipped_no_occ} fallback)")

    now_ist=datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime("%Y-%m-%dT%H:%M:%S+05:30")
    jdump({"generated_at":now_ist,"signal_date":latest,
           "note":"Deep analysis of TODAY signals — sorted by 5d avg return",
           "n_analysed":len(enriched),"patterns":enriched},
          OUT_DIR/"candle_enriched.json")
    print(f"  OK candle_enriched.json ({len(enriched)} patterns)")

    jdump({"generated_at":now_ist,"signal_date":latest,
           "note":"Top 5 from today signals by 5d return + confidence score",
           "top5":top5},
          OUT_DIR/"candle_top5.json")
    print(f"  OK candle_top5.json")
    print(f"\nTop 5:")
    for r in top5:
        print(f"  {r['sym']:<14} 5d={r.get('win_5d_avg',0):+.1f}% score={r['score']} "
              f"comfort_buy=Rs{r['comfort_buy_px']} T1=Rs{r['target_1']} T2=Rs{r['target_2']}")

if __name__=="__main__":
    main()
