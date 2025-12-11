"""
S&P 500 AI Investor — Fast + Recommendations + AI Summaries + Portfolio View

What this Streamlit app does for you:
- Loads S&P 500 tickers from a local CSV (no network headaches)
- Fetches fast metrics (price, market cap, 1y return, P/E) for ALL tickers in parallel
- Scores companies and produces a ranked list (Buffett-style bias toward quality & value)
- Suggests a monthly allocation for a given invest amount (e.g. $5,000)
- Lets you download a CSV buy list you can execute in Fidelity
- Lets you enter your portfolio (tickers + shares) and see its current value & weights
- Integrates with OpenAI to generate an AI "company briefing" for any selected ticker

Requirements (requirements.txt):
    streamlit
    pandas
    numpy
    yfinance
    openai

Before running, set your OpenAI API key in the environment, e.g. on macOS/Linux:
    export OPENAI_API_KEY="sk-..."

Run with:
    streamlit run sp500_ai_investor.py
"""

import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI

st.set_page_config(page_title="S&P500 AI Investor", layout="wide")

# ------------ Global settings ------------
FAST_WORKERS = 20          # parallel threads for yfinance.fast_info
CACHE_TTL = 60 * 60        # 1 hour cache
DEFAULT_MONTHLY = 5000
TOP_RECS_DEFAULT = 5

# Lazy OpenAI client (only created when needed)
_client = None


def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from env
    return _client


# ------------ Data loading ------------

@st.cache_data(ttl=CACHE_TTL)
def load_sp500_list() -> pd.DataFrame:
    """
    Load S&P 500 tickers, names, and sector from a LOCAL CSV file.
    Avoids any internet dependency so the app doesn't hang.
    """
    csv_path = "sp500_constituents.csv"
    df = pd.read_csv(csv_path)

    # Normalize columns
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
    Uses yfinance.fast_info which is much quicker than full .info.
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
    except Exception as e:  # graceful failure
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
            # Single-index df, columns are tickers
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
    Buffett-ish simple scoring:
    - Prefer large market cap (stability)
    - Prefer positive 1y return (trend)
    - Prefer moderate or low P/E
    """
    score = 0.0

    mc = row.get("marketCap")
    if pd.notna(mc) and mc > 0:
        # log scale market cap so we don't blow up scores
        score += np.log1p(mc) / 10

    r1y = row.get("1y_return")
    if pd.notna(r1y):
        score += r1y * 10  # +10 points for +100% year, -10 for -100%

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

    if method == "Equal weight":
        df["alloc_usd"] = monthly_amount / len(df)
    elif method == "Top 5 equal":
        k = min(5, len(df))
        df["alloc_usd"] = 0.0
        df.iloc[:k, df.columns.get_loc("alloc_usd")] = monthly_amount / k
    else:  # Score-weighted
        w = df["score"] - df["score"].min() + 1
        w = w / w.sum()
        df["alloc_usd"] = w * monthly_amount

    df["shares"] = np.floor(df["alloc_usd"] / df["price"].replace(0, np.nan)).fillna(0).astype(int)
    df["dollar_used"] = df["shares"] * df["price"]
    return df


# ------------ AI company briefing ------------

def fetch_company_snapshot(ticker: str) -> dict:
    """
    Grab a concise snapshot for AI to reason about (not a full 10-K but useful).
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
    Requires OPENAI_API_KEY in the environment.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return (
            "OPENAI_API_KEY is not set. "
            "Please set it in your environment before using AI summaries."
        )

    client = get_openai_client()

    snapshot = fetch_company_snapshot(ticker)

    # Build a compact text blob with our numeric metrics + snapshot
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
        # support formats: TICKER SHARES or TICKER:SHARES or TICKER,SHARES
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


# ------------ Streamlit UI ------------

sp500 = load_sp500_list()

st.title("S&P 500 AI Investor — Buffett Style Helper")
st.caption(
    "Fast S&P 500 screener + monthly buy suggestions + AI company briefings + portfolio view."
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
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

    # Clean numeric fields
    for col in ["price", "marketCap", "trailingPE", "1y_return"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["1y_return_pct"] = (df["1y_return"] * 100).round(2)

    # Compute scores
    df["score"] = df.apply(simple_quality_value_score, axis=1)
    df = df.sort_values("score", ascending=False)

    # Sector filter
    sectors = ["All"] + sorted(df["Sector"].fillna("Unknown").unique())
    sector_sel = st.selectbox("Filter by sector (or All)", options=sectors)
    if sector_sel != "All":
        df_filtered = df[df["Sector"] == sector_sel].copy()
    else:
        df_filtered = df.copy()

    st.subheader("Ranked companies (higher score = better mix of size, trend, valuation)")
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

    # --------- Monthly recommendations & allocation ---------
    st.markdown("---")
    st.subheader("Monthly Top Picks & Allocation")

    alloc_df = build_allocation(df_filtered, monthly_amount, top_recs, allocation_method)
    st.write(
        f"Suggested allocation for **${monthly_amount:,.0f}** this month across **{len(alloc_df)}** stocks."
    )

    st.dataframe(
        alloc_df[["Ticker", "Name", "price", "score", "alloc_usd", "shares", "dollar_used"]]
    )

    total_used = alloc_df["dollar_used"].sum()
    st.write(
        f"Total used: **${total_used:,.2f}** (cash left: ${monthly_amount - total_used:,.2f})"
    )

    csv_orders = alloc_df[
        ["Ticker", "Name", "price", "alloc_usd", "shares", "dollar_used"]
    ].to_csv(index=False)
    st.download_button(
        "Download CSV buy list (for Fidelity)",
        data=csv_orders,
        file_name="sp500_monthly_buy_list.csv",
    )

    # --------- AI company briefing ---------
    st.markdown("---")
    st.subheader("AI Company Briefing (Buffett-style)")

    tickers_for_brief = df_filtered["Ticker"].head(top_n_display).tolist()
    ticker_choice = st.selectbox("Choose a company to analyze", tickers_for_brief)
    if ticker_choice:
        row = df[df["Ticker"] == ticker_choice].iloc[0]
        name = row.get("Name", ticker_choice)
        st.write(f"**Selected:** {ticker_choice} — {name}")
        if st.button("Generate AI briefing for this company"):
            with st.spinner("Calling OpenAI to analyze this company..."):
                briefing = generate_ai_briefing(ticker_choice, name, row)
            st.markdown(briefing)

    # --------- Portfolio view ---------
    st.markdown("---")
    st.subheader("Your Portfolio Overview")
    st.write(
        "Paste holdings like: `AAPL 100` on one line, `MSFT 50` on another, etc. "
        "I will match them to S&P 500 data and show value & weights."
    )

    default_portfolio_text = """AAPL 100
MSFT 50
AMZN 40
"""
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

            st.write(f"Estimated portfolio market value: **${total_val:,.2f}**")
            st.dataframe(
                merged_port[["Ticker", "Name", "shares", "price", "value", "weight_pct"]]
            )

else:
    st.info("Click 'Load / Refresh metrics' in the sidebar to fetch S&P 500 metrics and start.")

st.markdown("---")
st.caption(
    "This tool is for research/education only and does not replace personalized financial advice. "
    "You still make the final decision and place orders in Fidelity yourself."
)
