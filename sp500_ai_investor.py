import os
import configparser
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI

st.set_page_config(page_title="S&P500 AI Investor", layout="wide")

FAST_WORKERS = 20
CACHE_TTL = 60 * 60
DEFAULT_MONTHLY = 5000
TOP_RECS_DEFAULT = 5


# ------------ Config / API keys from config.ini ------------

def load_config():
    """
    Load config from config.ini (if present).

    Expected format of config.ini in the same folder:

        [openai]
        api_key = sk-REPLACE_WITH_YOUR_KEY
    """
    cfg_path = "config.ini"
    if not os.path.exists(cfg_path):
        return
    config = configparser.ConfigParser()
    config.read(cfg_path)
    if "openai" in config:
        api_key = config["openai"].get("api_key")
        if api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key


load_config()

_client = None


def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI()  # will read OPENAI_API_KEY from env (possibly set by config.ini)
    return _client


# ------------ Data loading ------------

@st.cache_data(ttl=CACHE_TTL)
def load_sp500_list() -> pd.DataFrame:
    """
    Load S&P 500 tickers, names, and sector from a LOCAL CSV file.
    Expected file in same folder:
        sp500_constituents.csv
    """
    csv_path = "sp500_constituents.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Place sp500_constituents.csv next to this script."
        )
    df = pd.read_csv(csv_path)

    if "Symbol" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"Symbol": "Ticker"})
    if "Name" not in df.columns:
        df["Name"] = df["Ticker"]
    if "Sector" not in df.columns and "GICS Sector" in df.columns:
        df = df.rename(columns={"GICS Sector": "Sector"})
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    df["Ticker"] = df["Ticker"].astype(str).str.replace(".", "-", regex=False)
    return df[["Ticker", "Name", "Sector"]]


def fetch_fast_info(ticker: str) -> dict:
    """
    Fetch a small, fast subset of metrics for a single ticker.
    """
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", {}) or {}
        return {
            "Ticker": ticker,
            "price": fi.get("last_price")
            or fi.get("regularMarketPrice")
            or fi.get("previous_close"),
            "marketCap": fi.get("market_cap") or fi.get("marketCap"),
            "trailingPE": fi.get("trailing_pe") or fi.get("trailingPE"),
        }
    except Exception as e:
        return {"Ticker": ticker, "error": str(e)}


@st.cache_data(ttl=CACHE_TTL)
def bulk_fetch_metrics(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch fast metrics in parallel + 1-year returns via a bulk download.
    """
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=FAST_WORKERS) as ex:
        futures = {ex.submit(fetch_fast_info, t): t for t in tickers}
        for fut in as_completed(futures):
            results.append(fut.result())

    df_metrics = pd.DataFrame(results)

    # Bulk 1-year return from adj close
    try:
        price_df = yf.download(
            tickers=tickers,
            period="1y",
            group_by="ticker",
            threads=True,
            progress=False,
            timeout=60,
        )
        returns: dict[str, float] = {}
        if isinstance(price_df.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    s = price_df[t]["Adj Close"].dropna()
                    returns[t] = (s.iloc[-1] / s.iloc[0]) - 1 if len(s) > 1 else np.nan
                except Exception:
                    returns[t] = np.nan
        else:
            for t in tickers:
                if t in price_df.columns:
                    s = price_df[t].dropna()
                    returns[t] = (s.iloc[-1] / s.iloc[0]) - 1 if len(s) > 1 else np.nan
                else:
                    returns[t] = np.nan
    except Exception:
        returns = {t: np.nan for t in tickers}

    df_metrics["1y_return"] = df_metrics["Ticker"].map(returns)
    return df_metrics


# ------------ Scoring & allocation ------------

def simple_quality_value_score(row: pd.Series) -> float:
    """
    Simple Buffett-ish scoring:
    - Prefer large market cap (stability)
    - Prefer positive 1y return (trend)
    - Prefer moderate or low P/E
    """
    score = 0.0
    mc = row.get("marketCap")
    if pd.notna(mc) and mc > 0:
        score += np.log1p(mc) / 10
    r1y = row.get("1y_return")
    if pd.notna(r1y):
        score += r1y * 10
    pe = row.get("trailingPE")
    if pd.notna(pe) and pe > 0:
        if pe < 12:
            score += 5
        elif pe < 25:
            score += 2
        else:
            score -= 2
    return float(score)


def build_allocation(
    df: pd.DataFrame, monthly_amount: float, top_n: int, method: str
) -> pd.DataFrame:
    df = df.sort_values("score", ascending=False).head(top_n).copy()
    if len(df) == 0:
        return df

    if method == "Equal weight":
        df["alloc_usd"] = monthly_amount / len(df)
    elif method == "Top 5 equal":
        k = min(5, len(df))
        df["alloc_usd"] = 0.0
        df.iloc[:k, df.columns.get_loc("alloc_usd")] = monthly_amount / k
    else:
        w = df["score"] - df["score"].min() + 1
        w = w / w.sum()
        df["alloc_usd"] = w * monthly_amount
    df["shares"] = (
        np.floor(df["alloc_usd"] / df["price"].replace(0, np.nan))
        .fillna(0)
        .astype(int)
    )
    df["dollar_used"] = df["shares"] * df["price"]
    return df


# ------------ AI company briefing ------------

def fetch_company_snapshot(ticker: str) -> dict:
    """
    Grab a concise snapshot for AI to reason about.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}
    fields = {
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "longBusinessSummary": info.get("longBusinessSummary"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "dividendYield": info.get("dividendYield"),
        "profitMargins": info.get("profitMargins"),
        "returnOnEquity": info.get("returnOnEquity"),
        "freeCashflow": info.get("freeCashflow"),
        "totalDebt": info.get("totalDebt"),
    }
    return fields


def generate_ai_briefing(ticker: str, name: str, metrics_row: pd.Series) -> str:
    """
    Call OpenAI to produce a Buffett-style qualitative summary.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY is not set. Put it in config.ini under [openai] api_key=... "
            "or set it in your environment before using AI summaries."
        )

    client = get_openai_client()
    snapshot = fetch_company_snapshot(ticker)

    context_lines = [
        f"Ticker: {ticker}",
        f"Name: {name}",
        f"Price: {metrics_row.get('price')}",
        f"Market cap: {metrics_row.get('marketCap')}",
        f"1y return: {metrics_row.get('1y_return')}",
        f"Trailing P/E: {metrics_row.get('trailingPE')}",
        "--- Snapshot ---",
    ]
    for k, v in snapshot.items():
        context_lines.append(f"{k}: {v}")
    context_text = "\n".join(context_lines)

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an investment analyst helping a conservative retiree. "
                        "They like Warren Buffett style: high quality, durable, low-drama companies. "
                        "Write in clear, non-technical language."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Here is data about a company.\n"
                        "1) Briefly describe what the company does.\n"
                        "2) Summarize strengths.\n"
                        "3) Summarize key risks.\n"
                        "4) Give a simple verdict: 'wonderful company at fair/cheap/expensive price', "
                        "'OK company', or 'avoid for now'.\n\n"
                        + context_text
                    ),
                },
            ],
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


# ------------ Portfolio parsing ------------

def parse_portfolio_input(text: str) -> pd.DataFrame:
    """
    Parse lines like 'AAPL 100' or 'MSFT:50' into a holdings DataFrame.
    """
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in [" ", ":", ","]:
            if sep in line:
                parts = [p for p in line.replace(",", " ").split() if p]
                if len(parts) >= 2:
                    ticker = parts[0].upper()
                    try:
                        shares = float(parts[1])
                    except ValueError:
                        shares = 0.0
                    rows.append({"Ticker": ticker, "shares": shares})
                break
        else:
            continue
    if not rows:
        return pd.DataFrame(columns=["Ticker", "shares"])
    df = pd.DataFrame(rows)
    df = df.groupby("Ticker", as_index=False)["shares"].sum()
    return df


# ------------ Run Guide: Data Sources / Providers panel ------------

def render_data_sources_panel():
    """
    Transparency UI. Shows:
    - left: checkboxes for what the app uses today (and future ideas)
    - right: details table including Providers + Cost
    Note: these are informational right now; they don't switch the backend yet.
    """
    st.markdown("### Data Sources & Providers (Transparency)")
    st.caption("Checked = used by the app today (or optional). Unchecked = not implemented yet.")

    # Provider catalog (what exists in the world)
    providers = [
        {"id": "local_csv", "label": "Local CSV (sp500_constituents.csv)", "cost": "Free"},
        {"id": "yahoo", "label": "Yahoo Finance via yfinance", "cost": "Free"},
        {"id": "openai", "label": "OpenAI (AI analysis text)", "cost": "Varies (ChatGPT plan / API billed separately)"},
        {"id": "sec", "label": "SEC EDGAR (10-K/10-Q)", "cost": "Free"},
        {"id": "paid_feeds", "label": "Paid data feeds (FactSet/Bloomberg/Refinitiv/etc.)", "cost": "Paid $$$"},
        {"id": "hedge_fund", "label": "Proprietary hedge-fund APIs", "cost": "Not publicly available / $$$"},
        {"id": "hidden_openai_db", "label": "Hidden OpenAI stock databases", "cost": "❌ Not a real option"},
    ]

    # Sources used by the app (or future)
    sources = [
        {
            "id": "sp500_membership",
            "label": "S&P 500 membership (tickers, names, sectors)",
            "status": "Current (in use)",
            "default": True,
            "notes": "Loaded from local file: sp500_constituents.csv.",
            "providers_used": ["local_csv"],
        },
        {
            "id": "market_data",
            "label": "Market data (price, market cap, P/E, 1y history)",
            "status": "Current (in use)",
            "default": True,
            "notes": "Pulled using yfinance (Yahoo Finance).",
            "providers_used": ["yahoo"],
        },
        {
            "id": "ai_briefings",
            "label": "AI briefings (company explanation text)",
            "status": "Current (optional feature)",
            "default": True,
            "notes": "OpenAI does NOT provide stock prices; it summarizes the data we send it.",
            "providers_used": ["openai"],
        },
        {
            "id": "sec_filings",
            "label": "SEC EDGAR filings (10-K/10-Q) summarization",
            "status": "Future (not implemented)",
            "default": False,
            "notes": "Could be added later to summarize filings directly from the SEC.",
            "providers_used": ["sec"],
        },
        {
            "id": "paid_data",
            "label": "Professional paid feeds (better fundamentals coverage)",
            "status": "Future (not implemented)",
            "default": False,
            "notes": "Requires paid credentials and integration work.",
            "providers_used": ["paid_feeds"],
        },
        {
            "id": "hedge_fund_apis",
            "label": "Proprietary hedge-fund APIs",
            "status": "Future (not implemented)",
            "default": False,
            "notes": "Usually not available; would require legal access + credentials.",
            "providers_used": ["hedge_fund"],
        },
        {
            "id": "hidden_openai_db",
            "label": "Hidden OpenAI stock databases",
            "status": "Not a real option",
            "default": False,
            "notes": "Not a real data provider option. OpenAI doesn't offer a hidden stock DB.",
            "providers_used": ["hidden_openai_db"],
        },
    ]

    if "source_selection" not in st.session_state:
        st.session_state["source_selection"] = {s["id"]: s["default"] for s in sources}

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("#### Sources (select)")
        for s in sources:
            disabled = (s["status"] == "Not a real option")
            st.session_state["source_selection"][s["id"]] = st.checkbox(
                s["label"],
                value=bool(st.session_state["source_selection"].get(s["id"], s["default"])),
                disabled=disabled,
            )

    # Helpers for Providers and Cost columns
    provider_cost_map = {p["id"]: p["cost"] for p in providers}

    def providers_checklist(used_ids):
        lines = []
        for p in providers:
            mark = "✅" if p["id"] in used_ids else "⬜"
            lines.append(f"{mark} {p['label']}")
        return "\n".join(lines)

    def cost_summary(used_ids):
        # summarize unique costs for used providers
        costs = []
        seen = set()
        for pid in used_ids:
            c = provider_cost_map.get(pid, "")
            if c and c not in seen:
                seen.add(c)
                costs.append(c)
        return " | ".join(costs)

    with right:
        st.markdown("#### Details (including Providers + Cost)")
        selected = {k for k, v in st.session_state["source_selection"].items() if v}

        rows = []
        for s in sources:
            used = s.get("providers_used", [])
            rows.append(
                {
                    "Selected": "Yes" if s["id"] in selected else "No",
                    "Source": s["label"],
                    "Providers": providers_checklist(used),
                    "Cost": cost_summary(used),
                    "Status": s["status"],
                    "Notes": s["notes"],
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ------------ Streamlit UI ------------

sp500 = load_sp500_list()

st.title("S&P 500 AI Investor — Buffett Style Helper")
st.caption(
    "Fast S&P 500 screener + monthly buy suggestions + AI company briefings + portfolio view."
)

with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Go to",
        ["Dashboard", "Setup", "Run Guide"],
        index=0,
    )
    st.markdown("---")
    if section == "Dashboard":
        st.subheader("Settings")
        monthly_amount = st.number_input(
            "Monthly invest amount (USD)",
            value=DEFAULT_MONTHLY,
            step=500,
            min_value=0,
            format="%d",
        )
        top_n_display = st.slider("Show top N companies", 10, 100, 30, 5)
        top_recs = st.slider("Top picks for this month", 3, 15, TOP_RECS_DEFAULT, 1)
        allocation_method = st.selectbox(
            "Allocation method",
            ["Equal weight", "Score-weighted", "Top 5 equal"],
        )
        run_btn = st.button("Load / Refresh metrics")
        st.write("Last view: ", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    else:
        monthly_amount = DEFAULT_MONTHLY
        top_n_display = 30
        top_recs = TOP_RECS_DEFAULT
        allocation_method = "Equal weight"
        run_btn = False
        st.info("Use the tabs on the right to read the guide.")


# ---------- SECTION: SETUP GUIDE ----------

if section == "Setup":
    st.subheader("Setup — Installing and Preparing the App")

    tab_overview, tab_win, tab_mac, tab_vscode, tab_resources = st.tabs(
        ["Overview", "Windows Setup", "Mac Setup", "VS Code", "Resources"]
    )

    with tab_overview:
        st.markdown(
            """
### What this app is

This app is a personal S&P 500 investing assistant. It:

- Scans all S&P 500 companies
- Ranks them using simple quality + value rules
- Suggests how to invest your monthly amount (for example $5,000)
- Lets you export a buy list you can place at Fidelity
- Lets you paste your existing positions and see weights

To use it, you need:

1. Python 3.10+ installed  
2. A few Python packages: streamlit, pandas, numpy, yfinance, openai  
3. This script file: sp500_ai_investor.py  
4. The S&P 500 CSV file: sp500_constituents.csv  
5. (Optional) An OpenAI API key in config.ini for AI company summaries
"""
        )

    with tab_win:
        st.markdown(
            """
### Windows Setup (quick checklist)

1. Install Python 3 from the official website (check "Add Python to PATH").
2. Create a folder, for example: C:\\Users\\YOURNAME\\sp500_ai_investor
3. Put these files in that folder:
   - sp500_ai_investor.py
   - sp500_constituents.csv
4. Open VS Code, and open that folder.
5. In VS Code terminal, create a virtual environment:
   - python -m venv .venv
   - .venv\\Scripts\\activate
6. Install dependencies:
   - pip install streamlit pandas numpy yfinance openai
7. (Optional) Create config.ini in the same folder with:

   [openai]
   api_key = your-real-api-key

Then you can run the app (see Run Guide).
"""
        )

    with tab_mac:
        st.markdown(
            """
### macOS Setup (quick checklist)

1. Install Python 3 (via python.org or Homebrew).
2. Create a folder, for example: ~/sp500_ai_investor
3. Put these files in that folder:
   - sp500_ai_investor.py
   - sp500_constituents.csv
4. Open VS Code and open that folder.
5. In the terminal:

   - python3 -m venv .venv
   - source .venv/bin/activate

6. Install dependencies:

   - pip install streamlit pandas numpy yfinance openai

7. (Optional) Create config.ini with:

   [openai]
   api_key = your-real-api-key

Then run: streamlit run sp500_ai_investor.py
"""
        )

    with tab_vscode:
        st.markdown(
            """
### VS Code Notes

- Use File → Open Folder… to open the project.
- Use View → Terminal to open an integrated terminal.
- Activate your virtual environment every time you open the project.
- Run the app with: streamlit run sp500_ai_investor.py
- You do not need Node or NPM for this Python + Streamlit app.
"""
        )

    with tab_resources:
        st.markdown(
            """
### Extra Setup Resources

You can search for:

- "Install Python on Windows"
- "Install Python on macOS"
- "VS Code Python tutorial"
- "Streamlit getting started"

Follow these plus the checklists here, and you can run this S&P 500 investor app.
"""
        )


# ---------- SECTION: RUN GUIDE ----------

elif section == "Run Guide":
    st.subheader("Run Guide — How to Use This App Day to Day")

    (
        tab_run,
        tab_sources,
        tab_daily,
        tab_monthly,
        tab_buy,
        tab_yearend,
        tab_strat,
        tab_resources,
    ) = st.tabs(
        [
            "Running the App",
            "Data Sources",
            "Daily Use",
            "Monthly Investing Flow",
            "Buying in Fidelity",
            "Year-End Thoughts",
            "Strategies",
            "Resources",
        ]
    )

    with tab_run:
        st.markdown(
            """
### Running the app

1. Activate your virtual environment (if you use one).
2. In the project folder, run:
   - streamlit run sp500_ai_investor.py
3. Open the Local URL (usually http://localhost:8501) in your browser.

The Dashboard shows:

- Settings in the left sidebar
- S&P 500 snapshot
- Ranked companies
- Monthly allocation suggestion
- AI company briefing
- Portfolio view
"""
        )

    with tab_sources:
        render_data_sources_panel()

    with tab_daily:
        st.markdown(
            """
### Daily Use

- You normally do not need to trade daily.
- Opening the app daily is mainly for:
  - Learning about companies (AI briefing)
  - Checking your portfolio weights
  - Watching general trends

The strategy is long-term and low-drama, not day trading.
"""
        )

    with tab_monthly:
        st.markdown(
            """
### Monthly Investing Flow (example: $5,000 per month)

1. Go to the Dashboard.
2. Set "Monthly invest amount" (for example 5000).
3. Choose how many top picks you want (for example 5).
4. Choose the allocation method:
   - Equal weight
   - Score-weighted
   - Top 5 equal
5. Click "Load / Refresh metrics".
6. Look at "Monthly Top Picks & Allocation":
   - Tickers
   - Prices
   - Scores
   - Suggested dollars and shares
7. Download the CSV buy list if you like.
8. Enter the orders at your broker (e.g. Fidelity).

This gives you a rules-based monthly plan.
"""
        )

    with tab_buy:
        st.markdown(
            """
### Buying Stocks in Fidelity (high-level idea)

Screens change over time, but the flow is usually:

1. Log into Fidelity.
2. Go to the Trade / Stocks page.
3. For each ticker from the app:
   - Enter the symbol (for example AAPL).
   - Choose Buy.
   - Enter the number of shares suggested by the app.
   - Submit the order.
4. Repeat for each of the top picks.

After your trades fill, you can paste your positions into
the Portfolio section of the app and analyze weights.
"""
        )

    with tab_yearend:
        st.markdown(
            """
### Year-End Thoughts (not tax advice)

Key ideas for a long-term investor:

- You pay capital gains tax only when you sell at a profit.
- Holding good companies for many years keeps gains unrealized.
- Dividends may be taxable in the year you receive them.
- A Buffett-style approach usually prefers:
  - Fewer trades
  - Longer holding periods
  - High-quality businesses

For tax planning, always talk to a professional.
"""
        )

    with tab_strat:
        st.markdown(
            """
### Strategy — Buffett-flavored, simplified

This app is built around a few simple rules:

1. Bigger, more stable companies tend to score higher (market cap).
2. Positive 1-year returns help the score, very negative returns hurt it.
3. Moderate or low P/E is rewarded, very high P/E is penalized.
4. You invest a fixed amount each month into several good candidates.
5. AI is used to help you read about companies, not predict the future.

You still decide:

- Which businesses you like.
- How much you want to invest.
- When to place orders and when to hold cash.
"""
        )

    with tab_resources:
        st.markdown(
            """
### Extra Strategy Resources

Good topics to learn more about:

- Warren Buffett investing principles
- Margin of safety
- Dollar-cost averaging
- Behavioral biases in investing

The more you understand businesses and your own psychology,
the more helpful a tool like this becomes.
"""
        )


# ---------- SECTION: DASHBOARD (MAIN APP) ----------

else:
    st.subheader("S&P 500 universe snapshot")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(sp500.head(15))
    with col2:
        st.metric("Companies in index", len(sp500))

    if run_btn:
        all_tickers = sp500["Ticker"].tolist()
        with st.spinner("Fetching fast metrics for all S&P 500 companies..."):
            metrics_df = bulk_fetch_metrics(all_tickers)
            merged = metrics_df.merge(sp500, on="Ticker", how="left")
            st.session_state["metrics"] = merged
            st.success("Metrics loaded.")

    if "metrics" in st.session_state:
        df = st.session_state["metrics"].copy()

        for col in ["price", "marketCap", "trailingPE", "1y_return"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["1y_return_pct"] = (df["1y_return"] * 100).round(2)
        df["score"] = df.apply(simple_quality_value_score, axis=1)
        df = df.sort_values("score", ascending=False)

        sectors = ["All"] + sorted(df["Sector"].fillna("Unknown").unique())
        sector_sel = st.selectbox("Filter by sector (or All)", options=sectors)
        if sector_sel != "All":
            df_filtered = df[df["Sector"] == sector_sel].copy()
        else:
            df_filtered = df.copy()

        st.subheader("Ranked companies")
        st.dataframe(
            df_filtered[
                [
                    "Ticker",
                    "Name",
                    "Sector",
                    "price",
                    "marketCap",
                    "trailingPE",
                    "1y_return_pct",
                    "score",
                ]
            ].head(top_n_display)
        )

        st.markdown("---")
        st.subheader("Monthly Top Picks & Allocation")
        alloc_df = build_allocation(df_filtered, monthly_amount, top_recs, allocation_method)
        st.write(
            f"Suggested allocation for ${monthly_amount:,.0f} across {len(alloc_df)} stocks."
        )
        st.dataframe(
            alloc_df[
                ["Ticker", "Name", "price", "score", "alloc_usd", "shares", "dollar_used"]
            ]
        )
        total_used = alloc_df["dollar_used"].sum()
        st.write(
            f"Total used: ${total_used:,.2f} (cash left: ${monthly_amount - total_used:,.2f})"
        )

        csv_orders = alloc_df[
            ["Ticker", "Name", "price", "alloc_usd", "shares", "dollar_used"]
        ].to_csv(index=False)
        st.download_button(
            "Download CSV buy list (for broker)",
            data=csv_orders,
            file_name="sp500_monthly_buy_list.csv",
        )

        st.markdown("---")
        st.subheader("AI Company Briefing")
        tickers_for_brief = df_filtered["Ticker"].head(top_n_display).tolist()
        ticker_choice = st.selectbox("Choose a company to analyze", tickers_for_brief)
        if ticker_choice:
            row = df[df["Ticker"] == ticker_choice].iloc[0]
            name = row.get("Name", ticker_choice)
            st.write(f"Selected: {ticker_choice} — {name}")
            if st.button("Generate AI briefing for this company"):
                with st.spinner("Calling OpenAI to analyze this company..."):
                    briefing = generate_ai_briefing(ticker_choice, name, row)
                st.markdown(briefing)

        st.markdown("---")
        st.subheader("Your Portfolio Overview")
        st.write("Paste holdings like: AAPL 100 on one line, MSFT 50 on another.")
        default_portfolio_text = "AAPL 100\nMSFT 50\nAMZN 40\n"
        portfolio_text = st.text_area("Your positions", value=default_portfolio_text, height=120)
        if st.button("Analyze portfolio"):
            holdings_df = parse_portfolio_input(portfolio_text)
            if holdings_df.empty:
                st.warning("No valid positions parsed. Use format like 'AAPL 100'.")
            else:
                merged_port = holdings_df.merge(
                    df[["Ticker", "Name", "price"]], on="Ticker", how="left"
                )
                merged_port["value"] = merged_port["shares"] * merged_port["price"].fillna(0)
                total_val = merged_port["value"].sum()
                merged_port["weight_pct"] = np.where(
                    total_val > 0,
                    merged_port["value"] / total_val * 100,
                    0,
                ).round(2)
                st.write(f"Estimated portfolio market value: ${total_val:,.2f}")
                st.dataframe(
                    merged_port[
                        ["Ticker", "Name", "shares", "price", "value", "weight_pct"]
                    ]
                )
    else:
        st.info("Click 'Load / Refresh metrics' to fetch data and start.")

st.markdown("---")
st.caption(
    "This tool is for research/education only and does not replace personalized financial advice. "
    "You still make the final decision and place orders yourself."
)
