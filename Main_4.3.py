# Main_4.3.py — Combined Analysis for Thesis Section 4.3
# Combines:
# - Figure 18: Rolling 12-Month Beta (from excess returns)
# - Figure 19: Beta-Volatility Style Regime Maps (2x2 cross-plot)
# - Figure 20: Downside Distribution Scatter Plots (2x2 matrix)
# - Figure 21: Rolling 12-Month Downside Capture
# - Figure 22: Downside Capture Heatmap
# Features:
# - Rolling 12-month beta visualization with regime shading
# - Horizontal reference line at Beta = 1.0
# - Consistent fund colors
# - Interactive Plotly charts + Matplotlib/Seaborn static plots

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

st.set_page_config(page_title="Thesis", layout="wide")
st.title("")

# ============================================================
# REGIMES
# ============================================================
CUT_1 = pd.Timestamp("2020-02-01")
CUT_2 = pd.Timestamp("2022-01-01")

REGIME_RECTS = [
    ("Pre-COVID",      None,  CUT_1, "LightGreen",   0.18),
    ("COVID / Crisis", CUT_1,  CUT_2, "LightSalmon",  0.18),
    ("Post-COVID",     CUT_2,  None,  "LightSkyBlue", 0.18),
]

def add_regime_background(fig, x_min, x_max):
    for _, x0, x1, color, opacity in REGIME_RECTS:
        left = x_min if x0 is None else max(pd.Timestamp(x0), x_min)
        right = x_max if x1 is None else min(pd.Timestamp(x1), x_max)
        if left < right:
            fig.add_vrect(
                x0=left, x1=right,
                fillcolor=color, opacity=opacity,
                line_width=0,
                layer="below",
            )
    fig.add_vline(x=CUT_1, line_dash="dash", line_width=2)
    fig.add_vline(x=CUT_2, line_dash="dash", line_width=2)
    return fig

# ============================================================
# CONSISTENT FUND COLORS
# ============================================================
FUND_COLOR_MAP = {
    "ALCHWTU LX": "#8c5a0a",
    "GSCEQBA": "#ff2aa1",
    "MGGIX US": "#39ff14",
    "EMF US": "#ffd400",
}


# ============================================================
# HELPERS - BETA COMPUTATION
# ============================================================
def rolling_beta(rp: pd.Series, rm: pd.Series, window: int = 12) -> pd.Series:
    """
    CAPM rolling beta = Cov(Rp, Rm) / Var(Rm) over rolling window.
    From Figure_19 approach - cleaner implementation.
    """
    cov = rp.rolling(window).cov(rm)
    var = rm.rolling(window).var()
    return cov / var

def beta_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate full-sample beta from two arrays.
    From Figure_18 approach - for summary statistics.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    vx = np.var(x, ddof=1)
    if vx == 0 or not np.isfinite(vx):
        return np.nan
    return np.cov(x, y, ddof=1)[0, 1] / vx

def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dataframe: ensure Date column and sort."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def add_excess_and_beta(df: pd.DataFrame, rp_col: str, rm_col: str, rf_col: str, window: int):
    """
    Compute excess returns and rolling beta.
    Combines both approaches: pandas rolling for time series, beta_slope for full sample.
    """
    out = df.copy()
    out[rp_col] = pd.to_numeric(out[rp_col], errors="coerce")
    out[rm_col] = pd.to_numeric(out[rm_col], errors="coerce")
    out[rf_col] = pd.to_numeric(out[rf_col], errors="coerce")

    # Excess returns
    out["Fund_excess"] = out[rp_col] - out[rf_col]
    out["Benchmark_excess"] = out[rm_col] - out[rf_col]

    # Rolling beta using the cleaner approach from Figure_19
    out["Rolling_Beta"] = rolling_beta(out["Fund_excess"], out["Benchmark_excess"], window)

    # Full-sample beta for summary
    x = out["Benchmark_excess"]
    y = out["Fund_excess"]
    full_beta = beta_slope(x.values, y.values)

    return out, full_beta

# ===============================
# FIGURE 19 SPECIFIC: Configuration & Helpers
# ===============================
# Regime boundaries (thesis-safe, explicit)
PRE_END = pd.Timestamp("2020-03-01")          # Pre-COVID: < 2020-03-01
CRISIS_END = pd.Timestamp("2020-12-31")       # Crisis: 2020-03-01 ... 2020-12-31
# Post-COVID: > 2020-12-31

funds = {
    "MGGIX US": {
        "sheet": "NAV MGGIX US vs ACWI",
        "fund_ret": "Return MGGIX US, Rp",
        "bm_ret": "Return ACWI, Rm",
    },
    "EMF US": {
        "sheet": "NAV EMF US vs MSCI EM",
        "fund_ret": "Return EMF US, Rp",
        "bm_ret": "Return MSCI EM, Rm",
    },
    "GSCEQBA": {
        "sheet": "NAV GSCEQBA vs MSCI World Index",
        "fund_ret": "Return GSCEQBA, Rp",
        "bm_ret": "Return MSCI World Index",
    },
    "ALCHWTU LX": {
        "sheet": "NAV ALCHWTU LX v China all shar",
        "fund_ret": "Return ALCHWTU LX, Rp",
        "bm_ret": "Return China All Shares, Rm",
    },
}

# Match your regime color pattern request
REGIME_COLORS = {
    "Pre-COVID": "green",
    "COVID Crisis": "orange",
    "Post-COVID": "blue",
}

def add_regime(date_series: pd.Series) -> pd.Series:
    return np.where(
        date_series < PRE_END,
        "Pre-COVID",
        np.where(date_series <= CRISIS_END, "COVID Crisis", "Post-COVID"),
    )

@st.cache_data(show_spinner=False)
def load_and_compute(excel_path: str) -> dict:
    out = {}
    for name, cfg in funds.items():
        df = pd.read_excel(excel_path, sheet_name=cfg["sheet"])
        df["Date"] = pd.to_datetime(df["Date"])

        # Compute rolling metrics
        df["Beta"] = rolling_beta(df[cfg["fund_ret"]], df[cfg["bm_ret"]], window)
        df["Vol"] = df[cfg["fund_ret"]].rolling(window).std() * np.sqrt(12) * 100  # %

        df = df[["Date", "Beta", "Vol"]].dropna()
        df["Regime"] = add_regime(df["Date"])
        out[name] = df

    return out

def compute_global_axes(data: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Now returns axis limits as:
    - x limits for Volatility
    - y limits for Beta
    """
    all_beta = pd.concat([d["Beta"] for d in data.values()], axis=0)
    all_vol = pd.concat([d["Vol"] for d in data.values()], axis=0)

    if all_beta.empty or all_vol.empty:
        raise ValueError("No valid (Beta, Vol) points were computed. Check input return columns.")

    # X = Volatility
    vol_min, vol_max = float(all_vol.min() * 0.95), float(all_vol.max() * 1.05)
    # Y = Beta
    beta_min, beta_max = float(all_beta.min() * 0.98), float(all_beta.max() * 1.02)

    return (vol_min, vol_max), (beta_min, beta_max)

def plot_2x2(data: dict, vol_lim: tuple[float, float], beta_lim: tuple[float, float]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (name, df) in zip(axes, data.items()):
        for regime, color in REGIME_COLORS.items():
            sub = df[df["Regime"] == regime]
            # SWAPPED: X = Vol, Y = Beta
            ax.scatter(sub["Vol"], sub["Beta"], alpha=0.75, label=regime)

        # Reference line: Beta = 1 is now a HORIZONTAL line
        ax.axhline(1, linestyle="--")

        ax.set_title(f"{name}")
        ax.set_xlim(vol_lim)
        ax.set_ylim(beta_lim)
        ax.set_xlabel("Rolling Volatility (12M ann., %)")
        ax.set_ylabel("Rolling Beta (12M)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

# ===============================
# FIGURES 20-22 SPECIFIC: Downside Capture Configuration & Helpers
# ===============================
# Rolling DC settings
ROLLING_WINDOW_MONTHS = 12
MIN_DOWN_MONTHS = 3

# Fund colors for Figures 20-22
FUND_COLORS_DC = {
    "MGGIX US": "#00FF2A",
    "EMF US": "#FFEA00",
    "GSCEQBA": "#FF00A8",
    "ALCHWTU LX": "#8B5A00",
}

REGIME_COLOR_MAP_DC = {
    "Pre-COVID": "LightGreen",
    "COVID / Crisis": "LightSalmon",
    "Post-COVID": "LightSkyBlue",
}

def regime_label_from_date(d: pd.Timestamp) -> str:
    """Assign regime label based on date for DC figures."""
    if d < CUT_1:
        return "Pre-COVID"
    if d < CUT_2:
        return "COVID / Crisis"
    return "Post-COVID"

def add_regime_background_plotly(fig: go.Figure, x_min: pd.Timestamp, x_max: pd.Timestamp) -> go.Figure:
    """Add regime background shading to Plotly figure."""
    for _, x0, x1, color, opacity in REGIME_RECTS:
        left = x_min if x0 is None else max(pd.Timestamp(x0), x_min)
        right = x_max if x1 is None else min(pd.Timestamp(x1), x_max)
        if left < right:
            fig.add_vrect(x0=left, x1=right, fillcolor=color, opacity=opacity, line_width=0, layer="below")
    fig.add_vline(x=CUT_1, line_dash="dash", line_width=2)
    fig.add_vline(x=CUT_2, line_dash="dash", line_width=2)
    return fig

def calculate_dc_excel_equivalent(df_slice: pd.DataFrame, min_down_months: int = 3) -> float:
    """
    Calculate Downside Capture equivalent to Excel formula:

    IF(
      COUNT(FILTER(Rm_window, Rm_window<0)) >= 3,
      ( GEOMEAN(FILTER(1+Rp_window, Rm_window<0)) - 1 )
      /
      ( GEOMEAN(FILTER(1+Rm_window, Rm_window<0)) - 1 )
      * 100,
      NA()
    )
    """
    sub = df_slice.dropna(subset=["Rp", "Rm"]).copy()
    down = sub[sub["Rm"] < 0][["Rp", "Rm"]]

    if len(down) < min_down_months:
        return np.nan

    one_plus_rp = 1.0 + down["Rp"].astype(float)
    one_plus_rm = 1.0 + down["Rm"].astype(float)

    # Mirror Excel GEOMITTEL domain: all inputs must be > 0
    if (one_plus_rp <= 0).any() or (one_plus_rm <= 0).any():
        return np.nan

    gm_rp = np.exp(np.log(one_plus_rp).mean())
    gm_rm = np.exp(np.log(one_plus_rm).mean())

    gm_ret_rp = gm_rp - 1.0
    gm_ret_rm = gm_rm - 1.0

    if np.isclose(gm_ret_rm, 0.0, atol=1e-12):
        return np.nan

    return (gm_ret_rp / gm_ret_rm) * 100.0

@st.cache_data
def load_downside_data(file_path: str) -> dict:
    """Load data for downside capture analysis."""
    sheets_dc = {
        "ALCHWTU LX": "NAV ALCHWTU LX v China all shar",
        "GSCEQBA": "NAV GSCEQBA vs MSCI World Index",
        "MGGIX US": "NAV MGGIX US vs ACWI",
        "EMF US": "NAV EMF US vs MSCI EM",
    }

    all_data = {}
    for fund_name, sheet in sheets_dc.items():
        df = pd.read_excel(file_path, sheet_name=sheet)
        clean = df.iloc[:, [0, 3, 4]].copy()
        clean.columns = ["Date", "Rp", "Rm"]
        clean["Date"] = pd.to_datetime(clean["Date"])
        all_data[fund_name] = clean
    return all_data

# ===============================
# FIGURE 23 SPECIFIC: Behaviour Consistency Matrices Configuration & Helpers
# ===============================
# File names for Figure 23 (36-month data files)
FILES_FIG23 = {
    "ALCHWTU LX": "4_2_Allianz_All_China_Equity_MSCI_CHINA_ALL_SHARES_TOTAL_36m.xlsx",
    "GSCEQBA": "4_2_Goldman_Sachs_Global_CORE_Equity_vs_Benchmark_MSCI_World_Index.xlsx",
    "MGGIX US": "4_2_Morgan_Stanley_Global_Opportunity_MGGIX_NAV_2019_2024_36m.xlsx",
    "EMF US": "4_2_Templeton_Emerging_Markets_&_MSCI_Emerging_Markets_36m.xlsx",
}

# Thesis regimes for Figure 23 (inclusive)
PRE_START_F23   = pd.Timestamp("2019-01-01")
PRE_END_F23     = pd.Timestamp("2020-01-31")
COVID_START_F23 = pd.Timestamp("2020-02-01")
COVID_END_F23   = pd.Timestamp("2021-12-31")
POST_START_F23  = pd.Timestamp("2022-01-01")
POST_END_F23    = pd.Timestamp("2024-12-31")

# For Pre-COVID carry-in peak history
PEAK_HISTORY_START = pd.Timestamp("2018-01-01")

REGIMES_F23 = ["Pre-COVID", "During-COVID", "Post-COVID"]
ORDER_F23 = REGIMES_F23

# VERY SATURATED finance diverging colormap
# Negative = red, Positive = green, Neutral = white
SATURATED_FINANCE = LinearSegmentedColormap.from_list(
    "saturated_finance",
    [
        "#67000d",   # very dark red (extremely negative)
        "#cb181d",   # strong red
        "#ffffff",   # neutral
        "#7FBF00",   # strong green
        "#4F7F00",   # very dark green (extremely positive)
    ],
    N=256
)

def assign_regime_f23(date: pd.Timestamp) -> str:
    """Assign regime for Figure 23."""
    d = pd.Timestamp(date)
    if PRE_START_F23 <= d <= PRE_END_F23:
        return "Pre-COVID"
    if COVID_START_F23 <= d <= COVID_END_F23:
        return "During-COVID"
    if POST_START_F23 <= d <= POST_END_F23:
        return "Post-COVID"
    return "Out-of-sample"

def find_nav_column(df: pd.DataFrame) -> str:
    """Prefer NAV/Price-like column; fallback to the 2nd column."""
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["nav", "price", "close"]):
            return c
    return df.columns[1]

def rolling_capm_alpha_beta_and_te(
    fund_excess: np.ndarray,
    bench_excess: np.ndarray,
    window: int = 12
):
    """
    Rolling OLS:
      fund_excess = alpha + beta * bench_excess + e

    Rolling Tracking Error (annualized):
      TE = sqrt(12) * std(active_return_window),
      where active return = fund_excess - bench_excess
    """
    n = len(fund_excess)
    alpha_m = np.full(n, np.nan, float)
    beta = np.full(n, np.nan, float)
    te_ann = np.full(n, np.nan, float)

    for i in range(window - 1, n):
        y = fund_excess[i - window + 1 : i + 1]
        x = bench_excess[i - window + 1 : i + 1]

        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 3:
            continue

        y = y[mask]
        x = x[mask]

        X = np.column_stack([np.ones_like(x), x])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        a, bb = float(b[0]), float(b[1])

        alpha_m[i] = a
        beta[i] = bb

        ar = y - x
        if len(ar) > 1:
            te_ann[i] = float(np.sqrt(12) * np.std(ar, ddof=1))

    return alpha_m, beta, te_ann

def annualize_monthly_alpha(alpha_monthly: np.ndarray) -> np.ndarray:
    """Matches Excel convention: (1 + alpha)^12 - 1."""
    return (1.0 + alpha_monthly) ** 12 - 1.0

def mdd_with_carryin_peak_for_pre_only(df: pd.DataFrame, nav_col: str) -> pd.Series:
    """
    MDD per regime with specific definition:
    - Pre-COVID: carry-in peak from 2018 history (PEAK_HISTORY_START .. PRE_START-1)
    - During-COVID: reset peak at regime start
    - Post-COVID: reset peak at regime start
    Returns negative MDD values.
    """
    x = df.sort_values("Date").copy()
    out = {}

    # --- Pre-COVID: carry-in peak from 2018 ---
    pre = x[(x["Date"] >= PRE_START_F23) & (x["Date"] <= PRE_END_F23)]
    if not pre.empty:
        hist = x[(x["Date"] >= PEAK_HISTORY_START) & (x["Date"] < PRE_START_F23)]
        start_peak = float(hist[nav_col].max()) if not hist.empty else float(pre[nav_col].iloc[0])

        nav = pre[nav_col].astype(float).to_numpy()
        running_peak = np.maximum.accumulate(np.maximum(nav, start_peak))
        dd = nav / running_peak - 1.0
        out["Pre-COVID"] = float(np.min(dd))
    else:
        out["Pre-COVID"] = np.nan

    # --- During-COVID: reset peak ---
    during = x[(x["Date"] >= COVID_START_F23) & (x["Date"] <= COVID_END_F23)]
    if not during.empty:
        nav = during[nav_col].astype(float).to_numpy()
        running_peak = np.maximum.accumulate(nav)
        dd = nav / running_peak - 1.0
        out["During-COVID"] = float(np.min(dd))
    else:
        out["During-COVID"] = np.nan

    # --- Post-COVID: reset peak ---
    post = x[(x["Date"] >= POST_START_F23) & (x["Date"] <= POST_END_F23)]
    if not post.empty:
        nav = post[nav_col].astype(float).to_numpy()
        running_peak = np.maximum.accumulate(nav)
        dd = nav / running_peak - 1.0
        out["Post-COVID"] = float(np.min(dd))
    else:
        out["Post-COVID"] = np.nan

    return pd.Series(out)

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize within each fund across regimes (column-wise)."""
    return (df - df.mean()) / df.std(ddof=0)

# ===============================
# FIGURE 24 SPECIFIC: Beta-Alpha Space Configuration & Helpers
# ===============================
ALPHA_AS_PERCENT = True  # True -> alpha plotted in % (multiply by 100)

# Visual settings for Figure 24
FUND_LABELS_F24 = {
    "NAV ALCHWTU LX v China all shar": "ALCHWTU LX",
    "NAV GSCEQBA vs MSCI World Index": "GSCEQBA",
    "NAV EMF US vs MSCI EM": "EMF US",
    "NAV MGGIX US vs ACWI": "MGGIX",
}

FUND_COLORS_F24 = {
    "ALCHWTU LX": "#8c5a0a",  # brown
    "GSCEQBA":    "#ff2aa1",  # magenta
    "EMF US":     "#ffd400",  # yellow
    "MGGIX":      "#39ff14",  # green
}

REGIME_MARKERS_F24 = {
    "Pre-COVID": "^",   # triangle
    "COVID": "s",       # square (legend label shows During-COVID)
    "Post-COVID": "o",  # circle
}

REGIME_DISPLAY_F24 = {
    "Pre-COVID": "Pre-COVID",
    "COVID": "During-COVID",
    "Post-COVID": "Post-COVID",
}

def dedupe_columns(cols) -> list:
    """Remove duplicate column names."""
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out

def normalize_to_decimal(s: pd.Series) -> pd.Series:
    """Normalize returns to decimal format if needed."""
    x = pd.to_numeric(s, errors="coerce")
    med = np.nanmedian(np.abs(x.values)) if len(x) else 0.0
    if np.isfinite(med) and med > 1.0:
        x = x / 100.0
    return x

def detect_date_col(df: pd.DataFrame) -> str:
    """Detect date column."""
    for c in df.columns:
        if "date" in c.lower():
            return c
    return df.columns[0]

def detect_return_cols(df: pd.DataFrame) -> tuple:
    """Detect fund and benchmark return columns."""
    fund_ret = None
    bench_ret = None

    for c in df.columns:
        lc = c.lower()
        if "return" in lc and "rp" in lc:
            fund_ret = c
            break

    for c in df.columns:
        lc = c.lower()
        if "return" in lc and "rm" in lc:
            bench_ret = c
            break

    if bench_ret is None:
        for c in df.columns:
            lc = c.lower()
            if "return" in lc and "rp" not in lc:
                bench_ret = c
                break

    if fund_ret is None or bench_ret is None:
        raise ValueError(
            "Return columns not found. Expected: 'Return <Fund>, Rp' and 'Return <Benchmark>, Rm'."
        )
    return fund_ret, bench_ret

def detect_rf_col(df: pd.DataFrame) -> str:
    """Detect risk-free rate column."""
    for c in df.columns:
        lc = c.lower()
        if "risk-free" in lc or "risk free" in lc or lc == "rf":
            return c
    return None

def excel_slope(y: pd.Series, x: pd.Series) -> float:
    """Calculate slope using Excel-equivalent method."""
    tmp = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).dropna()
    if len(tmp) < 2:
        return np.nan
    xv = tmp["x"].values
    var_x = np.var(xv, ddof=1)
    if var_x == 0:
        return np.nan
    cov_xy = np.cov(tmp["x"].values, tmp["y"].values, ddof=1)[0, 1]
    return float(cov_xy / var_x)

def excel_intercept(y: pd.Series, x: pd.Series, slope: float) -> float:
    """Calculate intercept using Excel-equivalent method."""
    tmp = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).dropna()
    if len(tmp) < 2 or not np.isfinite(slope):
        return np.nan
    return float(tmp["y"].mean() - slope * tmp["x"].mean())

def regime_mask_month(
    dates: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series:
    """Create regime mask comparing by month (Period('M'))."""
    p = dates.dt.to_period("M")
    start_p = pd.Timestamp(start_date).to_period("M")
    if end_date is None:
        return p >= start_p
    end_p = pd.Timestamp(end_date).to_period("M")
    return (p >= start_p) & (p <= end_p)

def add_fund_regime_legend(ax):
    """Add fund and regime legend to the plot."""
    fund_order = ["ALCHWTU LX", "GSCEQBA", "EMF US", "MGGIX"]
    fund_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=FUND_COLORS_F24[f],
            markeredgecolor="#333333",
            markersize=9,
            label=f,
        )
        for f in fund_order
        if f in FUND_COLORS_F24
    ]

    regime_order = ["Pre-COVID", "COVID", "Post-COVID"]
    regime_handles = [
        Line2D(
            [0],
            [0],
            marker=REGIME_MARKERS_F24[r],
            linestyle="None",
            markerfacecolor="grey",
            markeredgecolor="#333333",
            markersize=9,
            label=REGIME_DISPLAY_F24[r],
        )
        for r in regime_order
    ]

    ax.legend(handles=fund_handles + regime_handles, loc="upper right", frameon=True)

def plot_beta_alpha(summary: pd.DataFrame):
    """Create Beta-Alpha space plot with regime transitions."""
    fig, ax = plt.subplots(figsize=(8, 8))
    order = ["Pre-COVID", "COVID", "Post-COVID"]

    ax.axvline(1.0, lw=1, linestyle="--", color="black")
    ax.axhline(0.0, lw=1, linestyle="-", color="black")

    for sheet_name, g in summary.groupby("Fund"):
        fund_label = FUND_LABELS_F24.get(sheet_name, sheet_name.replace("NAV ", ""))
        color = FUND_COLORS_F24.get(fund_label, "C0")

        g = g.set_index("Regime").reindex(order).reset_index()
        if g["Beta"].isna().any() or g["Alpha"].isna().any():
            continue

        xs = g["Beta"].values
        ys = g["Alpha"].values

        # light grey path line behind markers
        ax.plot(xs, ys, lw=0.8, color="black", alpha=0.95, zorder=1)

        # arrows
        arrow_kw = dict(
            arrowstyle="-|>",
            lw=0.8,
            color="black",
            shrinkA=7,
            shrinkB=7,
            mutation_scale=9,
        )
        ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]), arrowprops=arrow_kw, zorder=2)
        ax.annotate("", xy=(xs[2], ys[2]), xytext=(xs[1], ys[1]), arrowprops=arrow_kw, zorder=2)

        # markers
        for i, reg in enumerate(order):
            ax.scatter(
                xs[i],
                ys[i],
                marker=REGIME_MARKERS_F24[reg],
                s=140,
                facecolor=color,
                edgecolor="#333333",
                linewidth=1.1,
                zorder=3,
            )

    # quadrant labels
    ax.text(0.02, 0.78, "Defensive Skill", transform=ax.transAxes, fontsize=7, alpha=0.9)
    ax.text(0.42, 0.78, "Active Risk-Taking", transform=ax.transAxes, fontsize=7, alpha=0.9)
    ax.text(0.01, 0.06, "Ineffective Defence", transform=ax.transAxes, fontsize=7, alpha=0.9)
    ax.text(0.70, 0.06, "Risk Escalation", transform=ax.transAxes, fontsize=7, alpha=0.9)

    ax.set_xlabel("Market Beta")

    if ALPHA_AS_PERCENT:
        ax.set_ylabel("Annualized Alpha (%)")
    else:
        ax.set_ylabel("Annualized Alpha (decimal)")

    ax.set_title("")
    ax.grid(True, alpha=0.25, linestyle="--")

    add_fund_regime_legend(ax)
    return fig

# ============================================================
# SETTINGS + DATA PATH
# ============================================================
window = 12  # static 12-month rolling window

file_path = "Fund vs Benchmark_monthly.xlsx"
st.caption(f"")

# Sheets + columns configuration
SHEETS = [
    ("ALCHWTU LX", "NAV ALCHWTU LX v China all shar", "Return ALCHWTU LX, Rp", "Return China All Shares, Rm"),
    ("GSCEQBA",    "NAV GSCEQBA vs MSCI World Index", "Return GSCEQBA, Rp",    "Return MSCI World Index"),
    ("MGGIX US",   "NAV MGGIX US vs ACWI",            "Return MGGIX US, Rp",   "Return ACWI, Rm"),
    ("EMF US",     "NAV EMF US vs MSCI EM",           "Return EMF US, Rp",     "Return MSCI EM, Rm"),
]
rf_col = "Risk-free"


# ============================================================
# LOAD + COMPUTE
# ============================================================
results = {}

for fund_name, sheet_name, rp_col, rm_col in SHEETS:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = standardize_df(df)

    needed = [rp_col, rm_col, rf_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Sheet '{sheet_name}' missing columns: {missing}")
        st.stop()

    computed, full_beta = add_excess_and_beta(df, rp_col, rm_col, rf_col, window)

    results[fund_name] = {
        "data": computed,
        "full_beta": full_beta,
        "rp_col": rp_col,
        "rm_col": rm_col,
        "sheet": sheet_name,
    }

# ============================================================
# Figure 18. Rolling 12-Month Beta
# ============================================================

st.header("Figure 18. Rolling 12-Month Beta")

fig_roll = go.Figure()
all_idx = None

for fund_name, obj in results.items():
    s = obj["data"]["Rolling_Beta"].dropna()
    if s.empty:
        continue

    fig_roll.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=fund_name,
            line=dict(color=FUND_COLOR_MAP.get(fund_name)),
            hovertemplate="%{x|%Y-%m-%d}<br>Rolling Beta: %{y:.4f}<extra></extra>",
        )
    )
    all_idx = s.index if all_idx is None else all_idx.union(s.index)

if all_idx is None or len(all_idx) == 0:
    st.warning("No rolling beta values available (check data length).")
else:
    x_min, x_max = all_idx.min(), all_idx.max()

    # Reference line at Beta = 1
    fig_roll.add_hline(
        y=1.0,
        line_dash="dot",
        line_width=2,
        annotation_text="β = 1.0",
        annotation_position="top left",
    )

    fig_roll.update_xaxes(range=[x_min, x_max], hoverformat="%Y-%m-%d")
    fig_roll = add_regime_background(fig_roll, x_min, x_max)

    fig_roll.update_layout(
        xaxis_title="Date",
        yaxis_title="Beta",
        hovermode="x unified",
        height=520,
        margin=dict(l=40, r=40, t=70, b=40),
        legend_title="Funds",
    )
    st.plotly_chart(fig_roll, use_container_width=True)

# ===============================
# Figure 19. Beta-Volatility Style Regime Maps
# ===============================
st.header("Figure 19. Beta-Volatility Style Regime Maps ")

with st.spinner("Loading Excel + computing rolling beta/volatility..."):
    data = load_and_compute(file_path)

(vol_lim, beta_lim) = compute_global_axes(data)
fig = plot_2x2(data, vol_lim, beta_lim)

st.pyplot(fig, clear_figure=False)

# Optional: allow download as PNG (Streamlit-native)
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button(
    label="Download figure (PNG, 300 dpi)",
    data=buf.getvalue(),
    file_name="beta_volatility_crossplot_2x2_swapped_axes.png",
    mime="image/png",
)

# ===============================
# Load Downside Capture Data
# ===============================
try:
    data_dict_dc = load_downside_data(file_path)
except Exception as e:
    st.error(f"Error loading downside capture data: {e}")
    st.stop()

# ===============================
# Figure 20. Downside Distribution Scatter Plots
# ===============================
st.header("Figure 20. Downside Distribution Scatter Plots")

fig_m, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor="white")
axes = axes.flatten()

for i, (name, df) in enumerate(data_dict_dc.items()):
    ax = axes[i]
    down_only = df[df["Rm"] < 0].dropna(subset=["Rp", "Rm"])

    ax.scatter(down_only["Rm"], down_only["Rp"], color="#2c3e50", alpha=0.6, edgecolors="w", s=65)

    if not down_only.empty:
        limit = down_only["Rm"].min() * 1.15
        ax.plot([limit, 0.005], [limit, 0.005], color="#e74c3c", linestyle="--", linewidth=2, label="100% Capture")

    ax.set_title(f"Fund: {name}", fontweight="bold", fontsize=15)
    ax.set_xlabel("Benchmark Return ($R_m < 0$)")
    ax.set_ylabel("Fund Return ($R_p$)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.3)
    ax.legend(loc="upper left")

plt.tight_layout(pad=4)
st.pyplot(fig_m)

# ===============================
# Figure 21. Rolling 12-Month Downside Capture
# ===============================
st.header("Figure 21. Rolling 12-Month Downside Capture")

# Build one combined plotly chart (all funds), with regime shading + hover values
fig_dc = go.Figure()
all_dates_for_range = []

for fund_name, df in data_dict_dc.items():
    df_clean = df.dropna(subset=["Rp", "Rm"]).copy().sort_values("Date")
    w = ROLLING_WINDOW_MONTHS

    vals, dates, regimes = [], [], []
    if len(df_clean) >= w:
        for j in range(len(df_clean) - w + 1):
            window = df_clean.iloc[j : j + w]
            dc = calculate_dc_excel_equivalent(window, min_down_months=MIN_DOWN_MONTHS)

            if not np.isnan(dc):
                d = pd.Timestamp(df_clean.iloc[j + w - 1]["Date"])
                dates.append(d)
                vals.append(float(dc))
                regimes.append(regime_label_from_date(d))

    if dates:
        all_dates_for_range.append(pd.Series(dates))

        line_color = FUND_COLORS_DC.get(fund_name, "#444444")

        fig_dc.add_trace(
            go.Scatter(
                x=dates,
                y=vals,
                mode="lines+markers",
                name=fund_name,
                line=dict(color=line_color, width=2.6),
                marker=dict(size=4, color=line_color, line=dict(width=0.6, color="black")),
                hovertemplate=(
                    "Fund: %{fullData.name}"
                    "<br>Date: %{x|%Y-%m-%d}"
                    "<br>Rolling DC: %{y:.4f}%"
                    "<extra></extra>"
                ),
            )
        )

if not fig_dc.data:
    st.warning("Not enough data to compute rolling DC with the current window + minimum down-month rule.")
else:
    # Regime background shading based on the actual plotted x-range
    x_min = pd.concat(all_dates_for_range).min()
    x_max = pd.concat(all_dates_for_range).max()
    add_regime_background_plotly(fig_dc, x_min, x_max)

    # 100% benchmark line
    fig_dc.add_hline(y=100, line_dash="dash", line_width=2, line_color="red")

    fig_dc.update_layout(
        xaxis_title="Date",
        yaxis_title="Downside Capture (%)",
        hovermode="x unified",
        height=520,
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(title="Funds", orientation="v"),
    )
    fig_dc.update_xaxes(range=[x_min, x_max], hoverformat="%Y-%m-%d")

    st.plotly_chart(fig_dc, use_container_width=True)

# ===============================
# Figure 22. Downside Capture Heatmap (Regime Comparison) 
# ===============================
st.header("Figure 22. Downside Capture Heatmap (Regime Comparison) ")

# EXACT EXCEL ROW MAPPING
regime_indices = {
    "Pre-Covid": (1, 25),
    "During-Covid": (25, 48),
    "Post-Covid": (48, 84),
}

h_rows = []
for fund_name, df in data_dict_dc.items():
    res = {"Fund": fund_name}
    for r_name, (start, end) in regime_indices.items():
        res[r_name] = calculate_dc_excel_equivalent(df.iloc[start:end], min_down_months=MIN_DOWN_MONTHS)
    h_rows.append(res)

h_df = pd.DataFrame(h_rows).set_index("Fund")

fig_h, ax_h = plt.subplots(figsize=(8, 6))
sns.heatmap(h_df, annot=True, fmt=".2f", cmap="RdYlGn_r", center=100, linewidths=1.5, ax=ax_h)
st.pyplot(fig_h)

# ===============================
# Figure 23. Behaviour Consistency Matrices (Regime Comparison) 
# ===============================
st.header("Figure 23. Behaviour Consistency Matrices")

# Sidebar controls for Figure 23
with st.sidebar:
    st.subheader("Figure 23 Options")

    window_f23 = st.selectbox("Rolling window (months)", [12, 24, 36], index=0, key="window_f23")
    normalize_f23 = st.checkbox("Normalize within each fund (recommended)", value=True, key="normalize_f23")

    st.write("**Colors:** Saturated diverging: red (bad) → white → green (good).")

    # Recommended: symmetrical z-score range
    vmin_f23, vmax_f23 = st.slider("z-score range", -4.0, 4.0, (-2.5, 2.5), 0.1, key="zscore_f23")

    show_colorbar_f23 = st.checkbox("Show individual colorbars", value=True, key="colorbar_f23")
    annotate_f23 = st.checkbox("Annotate cells (raw values)", value=True, key="annotate_f23")
    show_tables_f23 = st.checkbox("Show regime-level tables", value=False, key="tables_f23")

# Compute matrices
required_cols_f23 = {"Date", "Fund Excess", "Benchmark Excess"}
matrices_f23 = {}

for fund, filename in FILES_FIG23.items():
    path = Path(filename)
    if not path.exists():
        st.error(f"File not found: {filename}")
        st.stop()

    df = pd.read_excel(path)
    missing = required_cols_f23 - set(df.columns)
    if missing:
        st.error(f"{fund}: missing required columns: {sorted(missing)}")
        st.stop()

    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"])
    x = x.sort_values("Date").reset_index(drop=True)
    x["Regime"] = x["Date"].apply(assign_regime_f23)

    nav_col = find_nav_column(x)

    # Rolling metrics on FULL sample
    fund_excess = x["Fund Excess"].astype(float).to_numpy()
    bench_excess = x["Benchmark Excess"].astype(float).to_numpy()

    alpha_m, beta, te_ann = rolling_capm_alpha_beta_and_te(fund_excess, bench_excess, window=window_f23)
    alpha_ann = annualize_monthly_alpha(alpha_m)

    x["_beta_roll"] = beta
    x["_alpha_roll_ann"] = alpha_ann
    x["_te_roll_ann"] = te_ann

    # Aggregate within regime months
    x_reg = x[x["Regime"].isin(REGIMES_F23)].copy()

    summary = x_reg.groupby("Regime").agg(
        **{
            "Aggressiveness (β)": ("_beta_roll", "median"),
            "Activeness (TE)": ("_te_roll_ann", "median"),
            "Value Add (α)": ("_alpha_roll_ann", "median"),
        }
    ).reindex(ORDER_F23)

    summary["Risk Control (MDD)"] = mdd_with_carryin_peak_for_pre_only(x, nav_col).reindex(ORDER_F23)

    matrices_f23[fund] = summary

if show_tables_f23:
    st.subheader("Regime-level tables")
    for fund, mat in matrices_f23.items():
        st.markdown(f"**{fund}**")
        st.dataframe(mat, use_container_width=True)

# Plot: 2×2 matrices with individual colorbars (saturated finance colors)
plt.rcParams.update({
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "font.size": 11,
})

fig_f23, axes_f23 = plt.subplots(2, 2, figsize=(14, 7), dpi=150)
axes_f23 = axes_f23.flatten()

for i, (fund, mat) in enumerate(matrices_f23.items()):
    ax = axes_f23[i]
    plot_df = zscore(mat) if normalize_f23 else mat

    im = ax.imshow(
        plot_df.values,
        cmap=SATURATED_FINANCE,  # saturated red-white-green
        vmin=vmin_f23,
        vmax=vmax_f23,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(fund, fontsize=12, pad=8)

    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)

    ax.set_xticks(range(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns, rotation=22, ha="right")

    # clean cell borders
    ax.set_xticks(np.arange(-.5, plot_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, plot_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate_f23:
        # annotate with raw values (not z-scores)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat.iloc[r, c]
                if pd.isna(val):
                    txt = ""
                else:
                    col = mat.columns[c]
                    if "Value Add" in col or "Drawdown" in col or "Activeness" in col:
                        txt = f"{val*100:.1f}%"
                    else:
                        txt = f"{val:.2f}"
                ax.text(c, r, txt, ha="center", va="center", fontsize=9, color="#111111")

    if show_colorbar_f23:
        cbar = fig_f23.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_linewidth(0.6)

for j in range(len(matrices_f23), 4):
    axes_f23[j].axis("off")

fig_f23.tight_layout()
st.pyplot(fig_f23, use_container_width=True)

# ===============================
# Figure 24. Beta-Alpha Space
# ===============================
st.header("Figure 24. Beta-Alpha Space")

# Load workbook for Figure 24
@st.cache_data
def load_workbook_f24(path: str) -> dict:
    """Load Excel workbook and return all sheets as dictionary."""
    xls = pd.ExcelFile(path)
    sheets = {}
    for sh in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sh)
        df.columns = dedupe_columns(df.columns)
        if df.shape[0] == 0 or df.shape[1] == 0:
            continue
        sheets[sh] = df
    return sheets

try:
    sheets_f24 = load_workbook_f24(file_path)
except Exception as e:
    st.error(f"Failed to read '{file_path}' for Figure 24: {e}")
    st.stop()

fund_sheets_f24 = [s for s in sheets_f24.keys() if s.lower().startswith("nav")]
if not fund_sheets_f24:
    st.error("No fund sheets found for Figure 24. Expected sheet names starting with 'NAV'.")
    st.stop()

# All fund sheets are used automatically (static)
selected_f24 = fund_sheets_f24

# Sidebar controls for Figure 24 (regime boundaries)
with st.sidebar:
    st.subheader("Figure 24 Options")
    st.caption("Regime boundaries (any day in month works; masking is monthly)")

    pre_start_f24 = st.date_input("Pre-COVID start", value=pd.to_datetime("2019-01-01").date(), key="pre_start_f24")
    pre_end_f24 = st.date_input("Pre-COVID end", value=pd.to_datetime("2020-01-31").date(), key="pre_end_f24")
    cov_start_f24 = st.date_input("COVID start", value=pd.to_datetime("2020-02-01").date(), key="cov_start_f24")
    cov_end_f24 = st.date_input("COVID end", value=pd.to_datetime("2021-12-31").date(), key="cov_end_f24")
    post_start_f24 = st.date_input("Post-COVID start", value=pd.to_datetime("2022-01-01").date(), key="post_start_f24")

regimes_f24 = [
    ("Pre-COVID", pd.to_datetime(pre_start_f24), pd.to_datetime(pre_end_f24)),
    ("COVID", pd.to_datetime(cov_start_f24), pd.to_datetime(cov_end_f24)),
    ("Post-COVID", pd.to_datetime(post_start_f24), None),
]

rows_f24 = []
notes_f24 = []

for sh in selected_f24:
    df = sheets_f24[sh].copy()

    try:
        date_col = detect_date_col(df)
        rp_col, rm_col = detect_return_cols(df)
    except Exception as e:
        notes_f24.append(f"[{sh}] {e}")
        continue

    rf_col = detect_rf_col(df)
    if rf_col is None:
        notes_f24.append(f"[{sh}] No Risk-free column found. Using Rf=0.")
        df["__Rf__"] = 0.0
        rf_col = "__Rf__"

    std = pd.DataFrame(
        {
            "Date": pd.to_datetime(df[date_col], errors="coerce"),
            "Rp": normalize_to_decimal(df[rp_col]),
            "Rm": normalize_to_decimal(df[rm_col]),
            "Rf": normalize_to_decimal(df[rf_col]).fillna(0.0),
        }
    ).dropna(subset=["Date"]).sort_values("Date")

    std["Excess_p"] = std["Rp"] - std["Rf"]
    std["Excess_m"] = std["Rm"] - std["Rf"]

    for reg_name, start, end in regimes_f24:
        sub = std.loc[regime_mask_month(std["Date"], start, end)].copy()

        beta = excel_slope(sub["Excess_p"], sub["Excess_m"])
        alpha_m = excel_intercept(sub["Excess_p"], sub["Excess_m"], beta)

        # alpha annualized (decimal), Excel-consistent
        alpha_ann = alpha_m * 12.0

        # ONLY requested transformation: show in %
        alpha_for_plot = alpha_ann * 100.0 if ALPHA_AS_PERCENT else alpha_ann

        rows_f24.append(
            {
                "Fund": sh,
                "Regime": reg_name,
                "Beta": beta,
                "Alpha": alpha_for_plot,
                "N": int(pd.DataFrame({"y": sub["Excess_p"], "x": sub["Excess_m"]}).dropna().shape[0]),
            }
        )

summary_f24 = pd.DataFrame(rows_f24)

if notes_f24:
    st.warning("Notes:\n- " + "\n- ".join(notes_f24))

if summary_f24.empty:
    st.error("No outputs computed for Figure 24. Check selected sheets and regime boundaries.")
    st.stop()

fig_f24 = plot_beta_alpha(summary_f24)
st.pyplot(fig_f24, clear_figure=True)
