#!/usr/bin/env python3
"""
candle_enriched_analysis.py
============================
Reads candle_best_patterns.json and for each top pattern, loads actual
daily price data to compute deep analysis:

  1. Gap analysis — does next day open above signal close? By how much?
  2. Entry opportunity — does next day's LOW dip below signal close?
     This answers: "can I buy cheaper even after a gap-up?"
  3. Pre-signal context — what was the trend in the 5/20/60/90 days before?
     Volume trend, price trend, distance from highs/lows
  4. Liquidity check — based on avg daily turnover, can I deploy Rs 2L?
  5. Best buy price, target, stop-loss for each pattern
  6. Pattern strength scoring — combining all above factors
  7. Top 5 stocks by combined score

Outputs → pattern_signals/:
  candle_enriched.json    — full deep analysis for all top patterns
  candle_top5.json        — top 5 stocks with all analysis + recommendations
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

LIQUIDITY_TARGET_RS = 200_000   # Rs 2 lakh per stock buy value
MIN_OCC_FOR_ANALYSIS = 3
TOP_N = 50   # analyse top N patterns from best_patterns.json

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

# ─────────────────────────────────────────────────────────────────────────────
# LOAD STOCK DATA — one stock at a time to save memory
# ─────────────────────────────────────────────────────────────────────────────
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

def load_stock(sym: str, trading_days: list) -> pd.DataFrame:
    """Load all daily data for one symbol across all trading days."""
    rows = []
    for ds in trading_days:
        y, m, _ = ds.split("-")
        path = DATA_DIR / "equity" / y / m / f"{ds}.csv"
        if not path.exists(): continue
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
            df.columns = df.columns.str.strip().str.strip('"').str.strip("'").str.upper()
            df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})
            needed = ["SYMBOL","OPEN","HIGH","LOW","CLOSE"]
            if not all(c in df.columns for c in needed): continue
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
                "tv": float(row["CLOSE"]) * vol,  # turnover estimate
            })
        except Exception:
            continue
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE CLASSIFICATION (same buckets as candle_patterns.py)
# ─────────────────────────────────────────────────────────────────────────────
def body_bkt(bp):
    if bp<0.3:  return "doji"
    if bp<1.0:  return "tiny"
    if bp<2.5:  return "small"
    if bp<5.0:  return "medium"
    return "large"

def wick_bkt(r):
    if pd.isna(r) or r<0.1:  return "none"
    if r<0.5:   return "tiny"
    if r<1.0:   return "small"
    if r<2.0:   return "medium"
    if r<4.0:   return "long"
    return "very_long"

def classify_key(o, h, l, c):
    body=abs(c-o); upper=h-max(o,c); lower=min(o,c)-l
    sb=body if body>0 else 0.0001
    bp=body/c*100 if c>0 else 0
    color="G" if c>=o else "R"
    return f"{color}_{body_bkt(bp)}_L{wick_bkt(lower/sb)}_U{wick_bkt(upper/sb)}"

# ─────────────────────────────────────────────────────────────────────────────
# DEEP ANALYSIS FOR ONE STOCK + CANDLE PATTERN
# ─────────────────────────────────────────────────────────────────────────────
def analyse_pattern(sym: str, candle_key: str, pat: dict, df: pd.DataFrame) -> dict:
    """
    For every historical occurrence of candle_key on sym, compute:
      - Gap: next_open / signal_close - 1 (% gap up/down at open)
      - Entry opp: did next day's LOW dip below signal close?
        If yes, by how much (entry_dip_pct). Means you can buy cheaper.
      - Max gain: (next_high - signal_close) / signal_close * 100
      - Next day range as % of signal close
      - Pre-signal trend (5d, 20d, 60d return before signal)
      - Pre-signal volume trend (vol on signal vs avg 20d before)
    """
    if df.empty or len(df) < 10:
        return {}

    # Classify each day
    df = df.copy().reset_index(drop=True)
    df["ckey"] = [classify_key(float(r.o),float(r.h),float(r.l),float(r.c))
                  for _, r in df.iterrows()]

    occurrences = []
    for i, row in df.iterrows():
        if row["ckey"] != candle_key: continue
        if i+1 >= len(df): continue   # no next day data

        sig_c  = float(row["c"])
        sig_v  = float(row["v"])
        nxt    = df.iloc[i+1]
        nxt_o  = float(nxt["o"])
        nxt_h  = float(nxt["h"])
        nxt_l  = float(nxt["l"])
        nxt_c  = float(nxt["c"])

        # Gap: how much did next day open vs signal close?
        gap_pct = round((nxt_o - sig_c) / sig_c * 100, 2)

        # Entry opportunity: did next day's LOW go below signal close?
        # If nxt_l < sig_c → you could have bought cheaper than signal close
        entry_dip  = nxt_l < sig_c
        entry_dip_pct = round((sig_c - nxt_l) / sig_c * 100, 2) if entry_dip else 0.0
        # Best buy price: if dip happened, you could buy at nxt_l; else at nxt_o
        best_entry = round(nxt_l if entry_dip else nxt_o, 2)
        best_entry_pct = round((best_entry - sig_c) / sig_c * 100, 2)

        # Maximum gain available next day (from best entry to nxt_h)
        max_gain_from_entry = round((nxt_h - best_entry) / best_entry * 100, 2) if best_entry > 0 else 0.0
        # Gain from signal close to next close
        nd_return = round((nxt_c - sig_c) / sig_c * 100, 2)
        # Gain from signal close to next day high (best possible)
        nd_max    = round((nxt_h - sig_c) / sig_c * 100, 2)

        # Pre-signal context (look back from signal day)
        pre5  = df.iloc[max(0,i-5):i]
        pre20 = df.iloc[max(0,i-20):i]
        pre60 = df.iloc[max(0,i-60):i]

        trend5  = round((sig_c - float(pre5.iloc[0]["c"])) / float(pre5.iloc[0]["c"]) * 100, 2) if len(pre5)>=1 else None
        trend20 = round((sig_c - float(pre20.iloc[0]["c"])) / float(pre20.iloc[0]["c"]) * 100, 2) if len(pre20)>=1 else None
        trend60 = round((sig_c - float(pre60.iloc[0]["c"])) / float(pre60.iloc[0]["c"]) * 100, 2) if len(pre60)>=1 else None

        # Volume context
        vol_avg20 = float(pre20["v"].mean()) if len(pre20)>=5 else 0.0
        vol_ratio = round(sig_v / vol_avg20, 2) if vol_avg20 > 0 else None

        # Turnover on signal day (Rs)
        turnover_rs = round(sig_c * sig_v)

        # Distance from 20-day high/low
        high20 = float(pre20["h"].max()) if len(pre20)>=1 else sig_c
        low20  = float(pre20["l"].min()) if len(pre20)>=1 else sig_c
        dist_from_high20 = round((sig_c - high20) / high20 * 100, 2)
        dist_from_low20  = round((sig_c - low20)  / low20  * 100, 2)

        occurrences.append({
            "date":              str(row["date"].date()),
            "sig_close":         round(sig_c, 2),
            "sig_volume":        int(sig_v),
            "turnover_rs":       turnover_rs,
            "vol_ratio_vs_20d":  vol_ratio,
            "gap_pct":           gap_pct,
            "entry_dip":         entry_dip,
            "entry_dip_pct":     entry_dip_pct,
            "best_entry_px":     best_entry,
            "best_entry_vs_close": best_entry_pct,
            "nd_return":         nd_return,
            "nd_max_gain":       nd_max,
            "max_gain_from_entry": max_gain_from_entry,
            "next_open":         round(nxt_o, 2),
            "next_high":         round(nxt_h, 2),
            "next_low":          round(nxt_l, 2),
            "next_close":        round(nxt_c, 2),
            "trend_5d_before":   trend5,
            "trend_20d_before":  trend20,
            "trend_60d_before":  trend60,
            "dist_from_high20":  dist_from_high20,
            "dist_from_low20":   dist_from_low20,
        })

    if len(occurrences) < MIN_OCC_FOR_ANALYSIS:
        return {}

    # Aggregate stats
    gaps      = [o["gap_pct"] for o in occurrences]
    dips      = [o["entry_dip_pct"] for o in occurrences]
    nd_rets   = [o["nd_return"] for o in occurrences]
    nd_maxes  = [o["nd_max_gain"] for o in occurrences]
    best_ents = [o["best_entry_vs_close"] for o in occurrences]
    max_from  = [o["max_gain_from_entry"] for o in occurrences]
    turnovers = [o["turnover_rs"] for o in occurrences]
    vol_ratios= [o["vol_ratio_vs_20d"] for o in occurrences if o["vol_ratio_vs_20d"]]

    n = len(occurrences)
    n_gap_up  = sum(1 for g in gaps if g > 0)
    n_dip     = sum(1 for o in occurrences if o["entry_dip"])

    # Liquidity: can we deploy Rs 2L?
    avg_turnover = round(sum(turnovers)/len(turnovers)) if turnovers else 0
    liquidity_ok = avg_turnover >= LIQUIDITY_TARGET_RS
    units_can_buy = int(LIQUIDITY_TARGET_RS / (occurrences[-1]["sig_close"] or 1)) if occurrences else 0

    # Buy/target/SL logic
    # Best buy = worst (highest) "best_entry_vs_close" pct across history
    # = if gap-up always happens, you buy at the gap open or slightly above close
    worst_entry_pct = round(max(best_ents), 2)   # worst case entry (most expensive)
    best_entry_pct_stat = round(min(best_ents), 2)  # best case (dip below close)
    avg_entry_pct = round(sum(best_ents)/len(best_ents), 2)

    # Targets based on actual historical outcomes
    min_nd_return = round(min(nd_rets), 2)   # lowest positive return (still positive)
    avg_nd_return = round(sum(nd_rets)/len(nd_rets), 2)
    max_nd_return = round(max(nd_rets), 2)
    min_nd_max    = round(min(nd_maxes), 2)  # lowest next-day-high gain
    avg_nd_max    = round(sum(nd_maxes)/len(nd_maxes), 2)

    # Stop loss: since all occurrences are positive, technically no stop needed.
    # But if next day's low was worst case X% below signal close, that's the
    # max adverse excursion before recovery.
    max_adverse   = round(max(dips), 2) if dips else 0.0  # max dip below close
    suggested_sl  = round(-max_adverse - 0.5, 1) if max_adverse > 0 else None
    # "No stop needed" if max adverse < 1% (stock never really dipped below close)
    sl_needed     = max_adverse > 1.0

    # Confidence factors
    pct_gap_up     = round(n_gap_up / n * 100, 1)
    pct_entry_dip  = round(n_dip / n * 100, 1)
    avg_vol_ratio  = round(sum(vol_ratios)/len(vol_ratios), 2) if vol_ratios else None
    avg_pre_trend  = [o["trend_20d_before"] for o in occurrences if o["trend_20d_before"] is not None]
    avg_trend_20d  = round(sum(avg_pre_trend)/len(avg_pre_trend), 2) if avg_pre_trend else None

    # Pattern strength score (0-100)
    score = 0
    score += 30  # base for being 100% win rate
    score += min(20, n * 2)  # more occurrences = higher confidence (max 20)
    score += min(15, len(set(o["date"][:4] for o in occurrences)) * 4)  # years
    if avg_vol_ratio and avg_vol_ratio > 1.5: score += 10  # high volume
    if pct_gap_up >= 80: score += 10  # consistent gap-up
    if not sl_needed: score += 10  # no adverse move
    if avg_turnover >= LIQUIDITY_TARGET_RS: score += 5  # liquid enough
    score = min(100, score)

    return {
        "sym":             sym,
        "candle_key":      candle_key,
        "desc":            pat.get("desc",""),
        "n_occurrences":   n,
        "years":           pat.get("years",[]),
        "score":           score,
        # Entry analysis
        "pct_gap_up":          pct_gap_up,
        "avg_gap_pct":         round(sum(gaps)/n, 2),
        "min_gap_pct":         round(min(gaps), 2),
        "max_gap_pct":         round(max(gaps), 2),
        "pct_entry_dip":       pct_entry_dip,
        "avg_dip_pct":         round(sum(dips)/n, 2),
        "max_dip_pct":         round(max(dips), 2) if dips else 0,
        "best_entry_vs_close": best_entry_pct_stat,
        "avg_entry_vs_close":  avg_entry_pct,
        "worst_entry_vs_close":worst_entry_pct,
        # Returns
        "nd_min_return":    pat.get("nd_min", min_nd_return),
        "nd_avg_return":    pat.get("nd_avg", avg_nd_return),
        "nd_max_return":    pat.get("nd_max", max_nd_return),
        "nd_min_high":      min_nd_max,
        "nd_avg_high":      avg_nd_max,
        "max_gain_from_best_entry": round(sum(max_from)/n, 2),
        # Recommendations
        "suggested_buy":    "At signal close or on next-open dip" if n_dip/n > 0.5 else "At gap-open",
        "target_conservative": round(avg_nd_return * 0.7, 1),  # 70% of avg
        "target_realistic":    round(avg_nd_return, 1),
        "target_optimistic":   round(min_nd_max, 1),  # min next-day high reached
        "sl_needed":           sl_needed,
        "max_adverse_excursion": max_adverse,
        "suggested_sl_pct":    suggested_sl,
        "sl_note":             (f"Max dip below entry was {max_adverse:.1f}% — consider SL at {suggested_sl:.1f}%"
                                if sl_needed else "No stop loss needed — never went below signal close in history"),
        # Liquidity
        "avg_turnover_rs":     avg_turnover,
        "liquidity_ok":        liquidity_ok,
        "units_buyable":       units_can_buy,
        "liquidity_note":      (f"Avg turnover Rs{avg_turnover:,} — can deploy Rs{LIQUIDITY_TARGET_RS:,} easily"
                                if liquidity_ok else
                                f"Low liquidity: avg Rs{avg_turnover:,} — may need 2-3 days to deploy Rs{LIQUIDITY_TARGET_RS:,}"),
        # Context
        "avg_vol_ratio_on_signal":  avg_vol_ratio,
        "avg_trend_20d_before":     avg_trend_20d,
        "pattern_strength_note":    _strength_note(avg_vol_ratio, avg_trend_20d, pct_gap_up, n),
        # Detailed occurrences (all rows, fully sortable in the UI)
        "occurrences":     occurrences,
    }


def _strength_note(vol_ratio, trend20, pct_gap_up, n):
    notes = []
    if vol_ratio and vol_ratio >= 2.0:
        notes.append(f"Volume was {vol_ratio:.1f}x avg on signal days (strong institutional interest)")
    elif vol_ratio and vol_ratio >= 1.3:
        notes.append(f"Volume was {vol_ratio:.1f}x avg on signal days (above-average interest)")
    if trend20 is not None and trend20 < -5:
        notes.append(f"Stock was down {abs(trend20):.1f}% in 20 days before — buying the dip")
    elif trend20 is not None and trend20 < 0:
        notes.append(f"Stock was weak ({trend20:.1f}%) in 20 days before signal — classic reversal setup")
    if pct_gap_up >= 80:
        notes.append(f"Gap-up on open in {pct_gap_up:.0f}% of times — market confirms signal overnight")
    if n >= 8:
        notes.append(f"Occurred {n} times — statistically very robust")
    elif n >= 5:
        notes.append(f"Occurred {n} times — good sample size")
    return ". ".join(notes) if notes else "Pattern is consistent — check individual occurrences for details"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Candle Enriched Analysis")
    print("="*65)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(MANIFEST) as f: manifest = json.load(f)
    tds = sorted(manifest.keys())
    print(f"  {len(tds)} trading days [{tds[0]} -> {tds[-1]}]")

    # Load best patterns
    bp_path = OUT_DIR / "candle_best_patterns.json"
    if not bp_path.exists():
        print("ERROR: candle_best_patterns.json not found. Run candle_patterns.py first.")
        sys.exit(1)
    with open(bp_path) as f: bp_data = json.load(f)
    best_pats = bp_data.get("patterns", [])
    print(f"  {len(best_pats)} best patterns to analyse")

    # Deduplicate: one entry per sym (take highest nd_min for each)
    sym_best = {}
    for pat in best_pats:
        sym = pat["sym"]
        if sym not in sym_best or pat["nd_min"] > sym_best[sym]["nd_min"]:
            sym_best[sym] = pat
    top_pats = sorted(sym_best.values(), key=lambda x: -x["nd_min"])[:TOP_N]
    print(f"  Analysing top {len(top_pats)} unique stocks")

    # Analyse each
    enriched = []
    for i, pat in enumerate(top_pats):
        sym = pat["sym"]
        ckey = pat["candle_key"]
        print(f"  [{i+1}/{len(top_pats)}] {sym} ({ckey})...", end="", flush=True)
        df = load_stock(sym, tds)
        if df.empty:
            print(" NO DATA")
            continue
        result = analyse_pattern(sym, ckey, pat, df)
        if not result:
            print(" INSUFFICIENT OCCURRENCES")
            continue
        enriched.append(result)
        print(f" score={result['score']} occ={result['n_occurrences']} "
              f"avg_ret={result['nd_avg_return']}%")

    # Sort by score descending
    enriched.sort(key=lambda x: (-x["score"], -x["nd_avg_return"]))
    top5 = enriched[:5]

    # Write outputs
    ist = timezone(timedelta(hours=5,minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%dT%H:%M:%S+05:30")

    jdump({"generated_at": now_ist,
           "latest_date": tds[-1],
           "n_analysed": len(enriched),
           "liquidity_target_rs": LIQUIDITY_TARGET_RS,
           "patterns": enriched},
          OUT_DIR / "candle_enriched.json")
    print(f"\n  OK candle_enriched.json ({len(enriched)} patterns)")

    jdump({"generated_at": now_ist,
           "latest_date": tds[-1],
           "note": "Top 5 stocks with 100%-win candle patterns, deepest analysis",
           "top5": top5},
          OUT_DIR / "candle_top5.json")
    print(f"  OK candle_top5.json")
    print(f"\nTop 5:")
    for r in top5:
        print(f"  {r['sym']:<14} score={r['score']}  nd_avg={r['nd_avg_return']:+.1f}%  "
              f"occ={r['n_occurrences']}  liq={'OK' if r['liquidity_ok'] else 'LOW'}")
    print("Done.")

if __name__ == "__main__":
    main()
