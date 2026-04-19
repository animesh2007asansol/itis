# 📈 NSE Market Data Pipeline

A fully automated, **100% free** GitHub-powered pipeline that:
- Downloads **7 categories** of NSE market data every trading day at **9 PM IST**
- Stores everything as versioned CSV files accessible via raw URLs
- Provides a **5-year historical backfill** with one button click
- Serves a live data portal at your GitHub Pages URL

**🌐 Live Portal:** https://animesh2007asansol.github.io/itis/

---

## Data collected

| # | Category | Key Columns |
|---|---|---|
| 1 | **Equity Bhav Copy** | SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE, TOTTRDQTY, TOTTRDVAL, TOTALTRADES, ISIN |
| 2 | **F&O Bhav Copy** | INSTRUMENT, SYMBOL, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW, CLOSE, SETTLE_PR, CONTRACTS, VAL_INLAKH, OPEN_INT, CHG_IN_OI |
| 3 | **Index Data** | Index Name, OPEN, HIGH, LOW, CLOSE, Points Change, Change%, Volume, Turnover |
| 4 | **Delivery Data** | SYMBOL, SERIES, DELIV_QTY, DELIV_VAL, TOTTRDQTY, TOTTRDVAL, DELIV_PER |
| 5 | **Corporate Actions** | SYMBOL, SERIES, EX_DATE, PURPOSE (split/bonus/dividend/rights), RATIO |
| 6 | **SME / Emerge** | Same as Equity Bhav for NSE Emerge platform |
| 7 | **Full Bhav (PR)** | 50+ columns comprehensive bhavcopy |

---

## File structure

```
data/
├── equity/YYYY/MM/YYYY-MM-DD.csv
├── fo/YYYY/MM/YYYY-MM-DD.csv
├── index/YYYY/MM/YYYY-MM-DD.csv
├── delivery/YYYY/MM/YYYY-MM-DD.csv
├── corporate_actions/
│   ├── YYYY/MM/YYYY-MM-DD.csv   ← daily snapshot
│   └── master.csv                ← all-time cumulative
├── sme/YYYY/MM/YYYY-MM-DD.csv
├── full_bhav/YYYY/MM/YYYY-MM-DD_*.csv
├── manifest.json                 ← index of all dates + fetch status
└── summary.json                  ← portal statistics
logs/YYYY-MM-DD.log
```

---

## Access data in your apps

```python
import pandas as pd, requests

BASE = "https://raw.githubusercontent.com/animesh2007asansol/itis/main/data"

# Equity OHLCV for any date
df = pd.read_csv(f"{BASE}/equity/2024/04/2024-04-10.csv")
print(df[["SYMBOL","OPEN","HIGH","LOW","CLOSE","TOTTRDQTY"]].head())

# All corporate actions master (splits, bonuses, dividends — all time)
master  = pd.read_csv(f"{BASE}/corporate_actions/master.csv")
splits  = master[master["PURPOSE"].str.contains("Split", case=False, na=False)]
bonuses = master[master["PURPOSE"].str.contains("Bonus", case=False, na=False)]

# Discover available dates
manifest = requests.get(f"{BASE}/manifest.json").json()
latest   = next(iter(manifest))            # most recent date
all_ok   = [d for d,v in manifest.items() if v.get("results",{}).get("equity")]
```

**URL pattern:**
```
https://raw.githubusercontent.com/animesh2007asansol/itis/main/data/{category}/{YYYY}/{MM}/{YYYY-MM-DD}.csv
```

---

## One-time setup (~5 minutes)

**1.** Create GitHub repo **`itis`** under `animesh2007asansol`

**2.** Push all these files to `main` (maintain folder structure)

**3.** Settings → Actions → General → Workflow permissions → **Read and write** → Save

**4.** Settings → Pages → Branch: `main` / `(root)` → Save

---

## Running the 5-year historical backfill

**Actions → "📚 NSE 5-Year Historical Backfill" → Run workflow**

Set `start_year = 2020`, leave `end_year` blank, click Run.

```
Setup job  →  generates year list [2020..2025]
                    ↓ parallel jobs
  ├─ Fetch 2020  (~230 days,  ~45 min)
  ├─ Fetch 2021  (~250 days,  ~50 min)
  ├─ Fetch 2022  (~250 days,  ~50 min)
  ├─ Fetch 2023  (~250 days,  ~50 min)
  └─ Fetch 2024/25  (~240 days, ~48 min)
                    ↓ collect job
  All artifacts merged → manifest rebuilt → single git push
```

**Total time: ~55 minutes** (parallel, limited by slowest year)

**Expected repo size:** 500 MB – 1.5 GB for 5 years of all categories.

After backfill, if some dates failed: **Actions → "🔁 Retry Failed Dates" → Run**

---

## Daily schedule

| IST | UTC | Purpose |
|---|---|---|
| 21:00 | 15:30 | Primary run |
| 21:30 | 16:00 | Retry 1 |
| 22:30 | 17:00 | Retry 2 |
| 23:00 | 17:30 | Final safety net |

---

## Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| 📈 NSE Daily Data Fetch | Auto (4×/day) + Manual | Daily OHLCV + actions |
| 📚 NSE 5-Year Backfill | Manual | Full historical download |
| 🗂️ NSE Historical Backfill | Manual | Custom date range |
| 🔁 Retry Failed Dates | Manual | Re-attempt failures |

---

## Robustness

- Browser cookie session warming (passes NSE anti-bot checks)
- 7-attempt retry with exponential back-off per URL
- Multiple CDN fallbacks (archives + www1)
- Session refresh every 50 dates during backfill
- Checkpoint file — resume from last failure
- Duplicate guard — skip dates already on disk
- Holiday detection via HEAD request
- Auto GitHub Issue on equity fetch failure

---

## Troubleshooting

**403 from NSE** — Wait 30 min and retry manually. NSE occasionally blocks IPs.  
**Missing dates** — Run "🔁 Retry Failed Dates" workflow.  
**Actions stopped** — GitHub disables schedules after 60 days of inactivity. Push any commit to re-enable.  
**Repo too large** — GitHub soft limit is 5 GB. If hit, archive old years to a separate repo.
