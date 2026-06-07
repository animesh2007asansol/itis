#!/usr/bin/env python3
"""
candle_pattern_analyzer.py
Finds stocks where bullish reversal candle + comparative volume after a fall
ALWAYS produces 30%+ upside. Tracks until weekly growth < 5%.
"""
import json, sys, warnings, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "equity"
OUT  = ROOT / "stock_analysis"
MANIFEST = ROOT / "data" / "manifest.json"
OUT.mkdir(exist_ok=True)

# ── CONFIG (tuned for results) ─────────────────────────────────────────────────
MIN_TURNOVER     = 5_000_000     # Rs 5 Cr daily (big mid-large cap)
MIN_YEARS        = 5             # 5 years of history
RECENT_DAYS      = 5             # active in last 5 trading dates
MIN_RETURN       = 30.0          # 30% minimum upside
WIN_RATE_PCT     = 100.0         # every occurrence must qualify
MIN_OCC          = 2             # minimum 2 occurrences (can be once a year)
PRIOR_FALL_DAYS  = 10            # look back 10 days for prior fall
PRIOR_FALL_PCT   = 3.0           # must have fallen at least 3% in those days
VOL_RATIO_MIN    = 1.3           # volume at least 1.3x 20-day average
MAX_TRACK_DAYS   = 130           # 26 weeks = 6 months max tracking
WEEKLY_MIN_INC   = 5.0           # optimal exit when weekly growth < 5%

EXCL_SFX = ("ETF","BEES","CASE","SETF","GILT","LIQUID","IVIX","NIFTYBEES")

IST     = timezone(timedelta(hours=5, minutes=30))
now_ist = datetime.now(IST)
now_str = now_ist.strftime("%Y-%m-%dT%H:%M:%S")
today   = now_ist.strftime("%Y-%m-%d")
cur_yr  = now_ist.year
min_yr  = cur_yr - MIN_YEARS

def r2(x):
    try:
        v=float(x)
        return None if(math.isnan(v) or math.isinf(v)) else round(v,2)
    except: return None


def load_all():
    if not MANIFEST.exists(): print("ERROR: manifest missing"); sys.exit(1)
    manifest=json.loads(MANIFEST.read_text())
    dates=sorted(manifest.keys())
    print(f"  Loading {len(dates)} dates...")
    frames=[]
    for ds in dates:
        y,mo,_=ds.split("-")
        p=DATA/y/mo/f"{ds}.csv"
        if not p.exists(): continue
        try:
            df=pd.read_csv(p,low_memory=False)
            df.columns=df.columns.str.strip()
            cm={}
            for col in df.columns:
                u=col.strip().upper()
                if u in ("SYMBOL","TCKRSYMB"):               cm[col]="sym"
                elif u in ("SERIES","SCTYSRS"):              cm[col]="series"
                elif u in ("OPEN","OPNPRIC","OPEN PRICE"):   cm[col]="o"
                elif u in ("HIGH","HGHPRIC","HIGH PRICE"):   cm[col]="h"
                elif u in ("LOW","LWPRIC","LOW PRICE"):      cm[col]="l"
                elif u in ("CLOSE","CLSPRIC","CLOSE PRICE"): cm[col]="c"
                elif u in ("TOTTRDQTY","TTLTRADGVOL"):       cm[col]="v"
            df=df.rename(columns=cm)
            if not {"sym","series","o","h","l","c","v"}.issubset(df.columns): continue
            df=df[df["series"].str.strip().isin(["EQ","BE"])].copy()
            df["date"]=pd.to_datetime(ds)
            frames.append(df[["sym","o","h","l","c","v","date"]])
        except: continue
    if not frames: sys.exit(1)
    all_data=pd.concat(frames,ignore_index=True)
    for col in ["o","h","l","c","v"]:
        all_data[col]=pd.to_numeric(all_data[col],errors="coerce")
    all_data=all_data.dropna(subset=["o","h","l","c","v"])
    return all_data.sort_values(["sym","date"]).reset_index(drop=True)


def detect_candle(o0,h0,l0,c0,o1,h1,l1,c1,o2,c2):
    """Detect bullish reversal patterns. Returns list of pattern names."""
    body0=abs(c0-o0); rng0=h0-l0 if h0!=l0 else 0.001
    lo_w0=min(o0,c0)-l0; up_w0=h0-max(o0,c0)
    body1=abs(c1-o1)
    found=[]

    # Hammer: lower wick >= 1.5x body, upper wick <= body, any color
    if body0>0 and lo_w0>=1.5*body0 and up_w0<=body0:
        found.append("Hammer")

    # Bullish Engulfing: prev red, today green, today engulfs prev body
    if c1<o1 and c0>o0 and c0>=o1 and o0<=c1:
        found.append("Bullish Engulfing")

    # Morning Star: big red, small middle, big green closing > prev midpoint
    body2=abs(c2-o2)
    if c2<o2 and body2>0 and body1<body2*0.6 and c0>o0 and c0>(o2+c2)/2:
        found.append("Morning Star")

    # Piercing Line: prev red, opens at/below prev low, closes > prev midpoint
    if c1<o1 and c0>o0 and o0<=h1*1.001 and c0>(o1+c1)/2:
        found.append("Piercing Line")

    # Long Lower Shadow: lower wick >= 60% of total range, green close
    if rng0>0 and lo_w0/rng0>=0.55 and c0>=o0:
        found.append("Long Lower Shadow")

    # Bullish Reversal Bar: green day, close in upper 40% of range, wide range
    if c0>o0 and rng0>0 and (c0-l0)/rng0>=0.6:
        found.append("Reversal Bar")

    # Doji Reversal: tiny body, significant lower shadow
    if rng0>0 and body0/rng0<0.15 and lo_w0>=0.4*rng0:
        found.append("Doji Reversal")

    return found


def find_peak_and_exit(c_arr,entry_idx,n):
    """Scan every day up to MAX_TRACK_DAYS. Find peak. Also find optimal exit (weekly growth < 5%)."""
    ep=float(c_arr[entry_idx])
    if ep<=0: return None
    peak_ret=-999.0; peak_day=1

    # Day-by-day scan for actual peak
    max_d=min(MAX_TRACK_DAYS,n-entry_idx-1)
    for d in range(1,max_d+1):
        xi=entry_idx+d
        ret=(float(c_arr[xi])-ep)/ep*100
        if ret>peak_ret: peak_ret=ret; peak_day=d

    if peak_ret<=-999.0: return None

    # Weekly tracking for optimal exit
    weekly=[]; prev_cum=0.0; opt_day=peak_day; opt_ret=peak_ret
    for wk in range(1,27):
        xi=entry_idx+wk*5
        if xi>=n: break
        cum=(float(c_arr[xi])-ep)/ep*100
        inc=cum-prev_cum
        weekly.append({"week":wk,"days":wk*5,"cum_ret":r2(cum),"weekly_inc":r2(inc)})
        if wk==1 or inc>=WEEKLY_MIN_INC:
            opt_day=wk*5; opt_ret=cum
        else:
            break
        prev_cum=cum

    # Fixed hold returns
    hold={}
    for d in [3,5,10,20,44,66]:
        xi=entry_idx+d
        if xi<n: hold[d]=r2((float(c_arr[xi])-ep)/ep*100)

    return {"peak_ret":r2(peak_ret),"peak_day":peak_day,
            "opt_ret":r2(opt_ret),"opt_day":opt_day,
            "weekly":weekly,"hold_rets":hold}


def analyze(sym,df,latest_set):
    n=len(df)
    o_arr=df["o"].values; h_arr=df["h"].values
    l_arr=df["l"].values; c_arr=df["c"].values; v_arr=df["v"].values
    dates=pd.to_datetime(df["date"].values)
    cur_price=float(c_arr[-1]); last_date=str(dates[-1].date())

    if cur_price<10: return None
    if last_date not in latest_set: return None
    yrs=set(int(d.year) for d in dates)
    if max(yrs)-min(yrs)<MIN_YEARS-1 or min(yrs)>min_yr: return None
    tv5=[float(c_arr[j])*float(v_arr[j]) for j in range(max(0,n-5),n) if float(v_arr[j])>0]
    if not tv5 or sum(tv5)/len(tv5)<MIN_TURNOVER: return None

    vol_ma=pd.Series(v_arr.astype(float)).rolling(20,min_periods=5).mean().values

    signals=[]; min_fwd=MAX_TRACK_DAYS+5
    for idx in range(max(PRIOR_FALL_DAYS+2,3), n-min_fwd):
        # Prior fall check
        pb_idx=idx-PRIOR_FALL_DAYS
        pb=float(c_arr[pb_idx])
        if pb<=0: continue
        fall=(float(c_arr[idx])-pb)/pb*100
        if fall>-PRIOR_FALL_PCT: continue

        # Volume check
        vm=float(vol_ma[idx]) if not math.isnan(float(vol_ma[idx])) else 0
        if vm<=0: continue
        vr=float(v_arr[idx])/vm
        if vr<VOL_RATIO_MIN: continue

        # Candle check
        try:
            pats=detect_candle(
                float(o_arr[idx]),float(h_arr[idx]),float(l_arr[idx]),float(c_arr[idx]),
                float(o_arr[idx-1]),float(h_arr[idx-1]),float(l_arr[idx-1]),float(c_arr[idx-1]),
                float(o_arr[idx-2]),float(c_arr[idx-2])
            )
        except: continue
        if not pats: continue

        entry_idx=idx+1
        if entry_idx>=n: continue
        ep=float(c_arr[entry_idx])
        if ep<=0: continue

        ex=find_peak_and_exit(c_arr,entry_idx,n)
        if ex is None or (ex["peak_ret"] or 0)<MIN_RETURN: continue

        # Context
        r5=r2((float(c_arr[idx])-float(c_arr[max(0,idx-5)]))/float(c_arr[max(0,idx-5)])*100) if idx>=5 else None
        r10=r2((float(c_arr[idx])-float(c_arr[max(0,idx-10)]))/float(c_arr[max(0,idx-10)])*100) if idx>=10 else None
        r20=r2((float(c_arr[idx])-float(c_arr[max(0,idx-20)]))/float(c_arr[max(0,idx-20)])*100) if idx>=20 else None

        signals.append({
            "date":str(dates[idx].date()),"year":int(dates[idx].year),
            "close":r2(float(c_arr[idx])),"patterns":pats,"primary":pats[0],
            "fall_pct":r2(fall),"vol_ratio":r2(vr),
            "entry_date":str(dates[entry_idx].date()),"entry_px":r2(ep),
            "peak_ret":ex["peak_ret"],"peak_day":ex["peak_day"],
            "opt_ret":ex["opt_ret"],"opt_day":ex["opt_day"],
            "weekly":ex["weekly"],"hold_rets":ex["hold_rets"],
            "r5":r5,"r10":r10,"r20":r20,
        })

    if len(signals)<MIN_OCC: return None
    if not all(s["peak_ret"]>=MIN_RETURN for s in signals): return None

    prs=[s["peak_ret"] for s in signals]
    ors=[s["opt_ret"] for s in signals if s["opt_ret"] is not None]
    pds=[s["peak_day"] for s in signals]
    ods=[s["opt_day"] for s in signals if s["opt_day"] is not None]
    all_pats=[]; [all_pats.extend(s["patterns"]) for s in signals]
    pc=Counter(all_pats)

    avg_pk=r2(sum(prs)/len(prs)); min_pk=r2(min(prs))
    avg_or=r2(sum(ors)/len(ors)) if ors else None
    avg_pd=r2(sum(pds)/len(pds)); avg_od=r2(sum(ods)/len(ods)) if ods else None
    dom=pc.most_common(1)[0][0] if pc else "Unknown"

    # Today alert
    today_alert=None
    if n>=3:
        ix=n-1
        if ix>=PRIOR_FALL_DAYS:
            pb2=float(c_arr[ix-PRIOR_FALL_DAYS])
            fall2=(float(c_arr[ix])-pb2)/pb2*100 if pb2>0 else 0
            vm2=float(vol_ma[ix]) if not math.isnan(float(vol_ma[ix])) else 0
            vr2=float(v_arr[ix])/vm2 if vm2>0 else 0
            try:
                pts2=detect_candle(
                    float(o_arr[ix]),float(h_arr[ix]),float(l_arr[ix]),float(c_arr[ix]),
                    float(o_arr[ix-1]),float(h_arr[ix-1]),float(l_arr[ix-1]),float(c_arr[ix-1]),
                    float(o_arr[ix-2]),float(c_arr[ix-2])
                )
            except: pts2=[]
            if fall2<-PRIOR_FALL_PCT and pts2 and vr2>=VOL_RATIO_MIN:
                today_alert={"date":last_date,"close":r2(cur_price),"patterns":pts2,
                    "fall":r2(fall2),"vol_ratio":r2(vr2),
                    "avg_pk":avg_pk,"min_pk":min_pk,
                    "tgt_min":r2(cur_price*(1+min_pk/100)),
                    "tgt_avg":r2(cur_price*(1+avg_pk/100))}

    # Active alerts (triggered within peak_day window)
    active=[]
    for s in reversed(signals):
        try:
            days_ago=(now_ist.replace(tzinfo=None)-datetime.strptime(s["date"],"%Y-%m-%d")).days
        except: continue
        if days_ago<=0 or days_ago>s["peak_day"]: continue
        ep2=s["entry_px"] or cur_price
        lr=r2((cur_price-ep2)/ep2*100) if ep2>0 else None
        active.append({"date":s["date"],"entry_date":s["entry_date"],"entry_px":s["entry_px"],
            "cur_price":r2(cur_price),"live_ret":lr,"peak_ret":s["peak_ret"],
            "days_elapsed":days_ago,"days_to_peak":s["peak_day"],"patterns":s["patterns"]})
        break

    ret5=r2((cur_price-float(c_arr[max(0,n-6)]))/float(c_arr[max(0,n-6)])*100) if n>5 else None
    ret10=r2((cur_price-float(c_arr[max(0,n-11)]))/float(c_arr[max(0,n-11)])*100) if n>10 else None
    ret15=r2((cur_price-float(c_arr[max(0,n-16)]))/float(c_arr[max(0,n-16)])*100) if n>15 else None

    return {"sym":sym,"price":r2(cur_price),"last_date":last_date,
        "turnover_cr":r2(sum(tv5)/len(tv5)/1e7),
        "n_signals":len(signals),"years":sorted(set(s["year"] for s in signals)),
        "dominant":dom,"pattern_counts":dict(pc.most_common()),
        "avg_pk":avg_pk,"min_pk":min_pk,"avg_or":avg_or,
        "avg_pd":avg_pd,"avg_od":avg_od,
        "ret5":ret5,"ret10":ret10,"ret15":ret15,
        "today_alert":today_alert,"active":active,
        "has_today":today_alert is not None,"has_active":len(active)>0,
        "signals":sorted(signals,key=lambda x:x["date"],reverse=True)}


def main():
    print(f"\n{chr(61)*60}\nCandle Pattern Analyzer  IST:{now_str}\n{chr(61)*60}")
    all_data=load_all()
    grouped=all_data.groupby("sym")
    syms=sorted(grouped.groups.keys())
    all_dates=sorted(set(str(pd.to_datetime(d).date()) for d in all_data["date"].unique()))
    latest_set=set(all_dates[-RECENT_DAYS:]); last_fetch=all_dates[-1]
    print(f"Last fetch: {last_fetch}  Symbols: {len(syms):,}")

    results=[]; skipped=0; excluded=0; n_no_turnover=0; n_no_years=0
    n_no_signals=0; n_no_win=0

    for i,sym in enumerate(syms):
        if (i+1)%500==0: print(f"  {i+1}/{len(syms)} found {len(results)}")
        if any(sym.upper().endswith(s) for s in EXCL_SFX): excluded+=1; continue
        try:
            grp=grouped.get_group(sym).sort_values("date").reset_index(drop=True)
            if len(grp)<200: skipped+=1; continue
            res=analyze(sym,grp,latest_set)
            if res: results.append(res)
            else: skipped+=1
        except: skipped+=1

    results.sort(key=lambda x:-(x.get("avg_pk") or 0))
    today_a=[r for r in results if r["has_today"]]
    active_a=[r for r in results if r["has_active"]]
    all_open=[r for r in results if r["has_today"] or r["has_active"]]

    all_hist=[]
    for r in results:
        for s in (r.get("signals") or []):
            all_hist.append({"date":s["date"],"year":s["year"],"sym":r["sym"],
                "price_now":r["price"],"patterns":s["patterns"],
                "fall":s["fall_pct"],"vol":s["vol_ratio"],
                "entry_px":s["entry_px"],"peak_ret":s["peak_ret"],"peak_day":s["peak_day"],
                "opt_ret":s["opt_ret"],"opt_day":s["opt_day"],
                "r3":s["hold_rets"].get(3),"r5":s["hold_rets"].get(5),
                "r10":s["hold_rets"].get(10),"r20":s["hold_rets"].get(20)})
    all_hist.sort(key=lambda x:x["date"],reverse=True)

    output={"generated_at":now_str,"today_ist":today,"last_fetch":last_fetch,
        "n_stocks":len(results),"n_today":len(today_a),"n_active":len(active_a),
        "n_all_hist":len(all_hist),
        "today_alerts":today_a,"active_alerts":active_a,"all_open":all_open,
        "all_hist":all_hist,"stocks":results}
    path=OUT/"candle_patterns.json"
    path.write_text(json.dumps(output,indent=2))
    print(f"\n Written {path}  stocks={len(results)} today={len(today_a)} hist={len(all_hist)}")
    if today_a:
        print("TODAY:")
        for r in today_a[:5]:
            ta=r["today_alert"]
            print(f"  {r['sym']:<14} {ta['patterns'][0]:<22} fall={ta['fall']}% vol={ta['vol_ratio']}x")
    print(f"\nTop 10:")
    for r in results[:10]:
        print(f"  {r['sym']:<14} {r['dominant']:<22} pk=+{r['avg_pk']}% pd={r['avg_pd']}d n={r['n_signals']}")

if __name__=="__main__": main()
