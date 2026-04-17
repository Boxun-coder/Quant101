from __future__ import annotations

import base64
import contextlib
import io
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
if VENDOR_DIR.exists():
    sys.path.append(str(VENDOR_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jinja2 import Template
from matplotlib.backends.backend_pdf import PdfPages

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    HTML = None
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_ERROR = exc

from src import config
from src.alphas import build_raw_alpha
from src.backtester import compute_metrics, run_backtest
from src.portfolio import build_target_weights


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    results = pd.read_parquet(config.RESULTS_MATRIX_FILE)
    ret = pd.read_parquet(config.RETURN_FILE)
    vol = pd.read_parquet(config.VOLUME_FILE)
    benchmark = pd.read_parquet(config.BENCHMARK_FILE)["sprtrn"]

    results.index = pd.to_datetime(results.index)
    ret.index = pd.to_datetime(ret.index)
    vol.index = pd.to_datetime(vol.index)
    benchmark.index = pd.to_datetime(benchmark.index)

    return results.sort_index(), ret.sort_index(), vol.sort_index(), benchmark.sort_index()


def _compute_period_metrics(results: pd.DataFrame, start: str | None = None, end: str | None = None) -> dict[str, float]:
    period = results.copy()
    if start is not None:
        period = period.loc[pd.Timestamp(start) :]
    if end is not None:
        period = period.loc[: pd.Timestamp(end)]
    return compute_metrics(period["daily_return"], period["turnover"])


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _format_num(value: float) -> str:
    return f"{value:.2f}"


def _finalize_figure(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _plot_equity_curve(results: pd.DataFrame) -> str:
    split_date = pd.Timestamp(config.OUT_OF_SAMPLE_START)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    equity = (1.0 + results["daily_return"].fillna(0.0)).cumprod()

    is_mask = equity.index < split_date
    oos_mask = equity.index >= split_date

    ax.plot(equity.index[is_mask], equity[is_mask], color="#1f77b4", linewidth=2.2, label="In-Sample")
    ax.plot(equity.index[oos_mask], equity[oos_mask], color="#ff7f0e", linewidth=2.2, label="Out-of-Sample")
    ax.axvline(split_date, color="#3d3d3d", linestyle="--", linewidth=1.4)
    ax.set_title("Cumulative Strategy Equity", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    return _finalize_figure(fig)


def _plot_drawdown(results: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.fill_between(results.index, results["drawdown"], 0.0, color="#c44e52", alpha=0.82)
    ax.plot(results.index, results["drawdown"], color="#8c2d2f", linewidth=1.1)
    ax.set_title("Drawdown Profile", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2)
    return _finalize_figure(fig)


def _compute_factor_attribution(ret: pd.DataFrame, vol: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    factors = build_raw_alpha(ret, vol)
    rows = []
    for name in ["reversal_5d", "volume_shock", "momentum_12m_1m"]:
        weights, _ = build_target_weights(factors[name], ret, benchmark)
        _, turnover, metrics = run_backtest(weights, ret)
        rows.append(
            {
                "factor": name.replace("_", " ").title(),
                "sharpe_ratio": metrics["sharpe_ratio"],
                "annualized_return": metrics["annualized_return"],
                "average_daily_turnover": turnover.mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot_factor_attribution(factor_stats: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    sns.barplot(
        data=factor_stats,
        x="factor",
        y="sharpe_ratio",
        hue="factor",
        palette=["#4c78a8", "#f58518", "#54a24b"],
        legend=False,
        ax=ax,
    )
    ax.axhline(0.0, color="#2f2f2f", linewidth=1.0)
    ax.set_title("Standalone Factor Sharpe Ratios", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(axis="y", alpha=0.2)
    return _finalize_figure(fig)


def _build_html(
    full_metrics: dict[str, float],
    is_metrics: dict[str, float],
    oos_metrics: dict[str, float],
    equity_chart: str,
    drawdown_chart: str,
    factor_chart: str,
    factor_stats: pd.DataFrame,
) -> str:
    template = Template(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Statistical Arbitrage Strategy Report</title>
  <style>
    @page { size: A4; margin: 0.55in; }
    body {
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: #1f2933;
      margin: 0;
      font-size: 11px;
      line-height: 1.45;
      background: #ffffff;
    }
    .page-break { page-break-before: always; }
    .hero {
      padding: 22px 24px;
      background: linear-gradient(135deg, #0b3954, #087e8b);
      color: white;
      border-radius: 14px;
      margin-bottom: 18px;
    }
    h1 { margin: 0 0 6px 0; font-size: 24px; }
    h2 {
      margin: 22px 0 8px 0;
      font-size: 16px;
      color: #0b3954;
      border-bottom: 2px solid #d9e2ec;
      padding-bottom: 4px;
    }
    h3 { margin: 16px 0 6px 0; font-size: 13px; color: #102a43; }
    p { margin: 0 0 10px 0; }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin: 12px 0 18px 0;
    }
    .card {
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      padding: 12px 14px;
      background: #f8fbfd;
    }
    .label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em; color: #627d98; }
    .value { font-size: 18px; font-weight: 700; color: #102a43; margin-top: 4px; }
    .columns {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 18px;
      align-items: start;
    }
    .note {
      background: #f0f4f8;
      border-left: 4px solid #087e8b;
      padding: 10px 12px;
      border-radius: 6px;
    }
    img { width: 100%; border: 1px solid #d9e2ec; border-radius: 10px; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 10.5px;
    }
    th, td {
      border-bottom: 1px solid #d9e2ec;
      padding: 8px 6px;
      text-align: left;
    }
    th { background: #f0f4f8; color: #243b53; }
    .footer { margin-top: 18px; color: #627d98; font-size: 9px; }
  </style>
</head>
<body>
  <div class="hero">
    <h1>Statistical Arbitrage Research Report</h1>
    <p>Market-neutral equity stat arb backtest across S&amp;P 500 constituents, January 2016 through December 2024.</p>
  </div>

  <div class="grid">
    <div class="card"><div class="label">Full Period Sharpe</div><div class="value">{{ full.sharpe_ratio }}</div></div>
    <div class="card"><div class="label">Annualized Return</div><div class="value">{{ full.annualized_return }}</div></div>
    <div class="card"><div class="label">Average Daily Turnover</div><div class="value">{{ full.average_daily_turnover }}</div></div>
  </div>

  <h2>Executive Summary &amp; Philosophy</h2>
  <p>This strategy is a market-neutral statistical arbitrage process designed to monetize transient mispricings inside a liquid large-cap universe. The alpha stack blends short-horizon mean reversion, volume-dislocation information, and medium-horizon momentum, then strips out broad market exposure before capital is allocated.</p>
  <p>The economic intuition is straightforward: large-cap stocks can temporarily overshoot due to liquidity shocks, crowding, or delayed information diffusion. By combining complementary cross-sectional signals and enforcing beta neutrality, the portfolio seeks to retain stock-specific opportunity while reducing dependence on systematic market direction.</p>

  <h2>Data Partitioning (IS vs OOS)</h2>
  <div class="columns">
    <div>
      <p><strong>In-Sample (2016-01 to 2020-12):</strong> used for factor discovery and parameter tuning.</p>
      <p><strong>Out-of-Sample (2021-01 to 2024-12):</strong> treated as the blind validation segment. No parameter tweaking occurred in this period.</p>
      <div class="note">
        The out-of-sample window is the key reality check: it measures whether the research process generalized after the design choices were fixed.
      </div>
    </div>
    <div>
      <table>
        <thead>
          <tr><th>Period</th><th>Ann. Return</th><th>Volatility</th><th>Sharpe</th><th>Max DD</th><th>Turnover</th></tr>
        </thead>
        <tbody>
          <tr><td>In-Sample</td><td>{{ insample.annualized_return }}</td><td>{{ insample.annualized_volatility }}</td><td>{{ insample.sharpe_ratio }}</td><td>{{ insample.max_drawdown }}</td><td>{{ insample.average_daily_turnover }}</td></tr>
          <tr><td>Out-of-Sample</td><td>{{ oos.annualized_return }}</td><td>{{ oos.annualized_volatility }}</td><td>{{ oos.sharpe_ratio }}</td><td>{{ oos.max_drawdown }}</td><td>{{ oos.average_daily_turnover }}</td></tr>
          <tr><td>Full Period</td><td>{{ full.annualized_return }}</td><td>{{ full.annualized_volatility }}</td><td>{{ full.sharpe_ratio }}</td><td>{{ full.max_drawdown }}</td><td>{{ full.average_daily_turnover }}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <h2>Backtest Assumptions</h2>
  <p><strong>T+1 Execution:</strong> signals formed at the close of Day T are executed at the close of Day T+1, which removes look-ahead bias from the backtest implementation.</p>
  <p><strong>Transaction Costs:</strong> the simulation deducts 5 basis points (0.05%) per unit of turnover directly from daily P&amp;L.</p>
  <p><strong>Slippage &amp; Shorting:</strong> the framework assumes full borrow availability across the S&amp;P 500 universe and negligible market impact because the strategy operates in highly liquid large-cap names.</p>

  <h2>Cumulative Returns</h2>
  <img src="data:image/png;base64,{{ equity_chart }}" alt="Cumulative returns chart">

  <div class="page-break"></div>

  <h2>Drawdown Analysis</h2>
  <p>The drawdown curve highlights the strategy's peak-to-trough risk profile. Because the portfolio is cross-sectionally de-meaned and beta-neutralized, drawdowns are primarily driven by stock-selection misspecification or temporary crowding rather than outright market direction.</p>
  <img src="data:image/png;base64,{{ drawdown_chart }}" alt="Drawdown chart">

  <h2>Factor Attribution</h2>
  <p>The standalone factor lens shows how each raw signal behaves independently before aggregation. This is useful for understanding diversification across the alpha stack and for diagnosing whether one leg dominates performance or turnover.</p>
  <img src="data:image/png;base64,{{ factor_chart }}" alt="Factor attribution chart">

  <table>
    <thead>
      <tr><th>Factor</th><th>Standalone Sharpe</th><th>Ann. Return</th><th>Avg. Daily Turnover</th></tr>
    </thead>
    <tbody>
      {% for row in factor_rows %}
      <tr>
        <td>{{ row.factor }}</td>
        <td>{{ row.sharpe_ratio }}</td>
        <td>{{ row.annualized_return }}</td>
        <td>{{ row.average_daily_turnover }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <div class="footer">
    Generated automatically from backtester artifacts in {{ clean_dir }}.
  </div>
</body>
</html>
        """
    )

    def metric_row(metrics: dict[str, float]) -> dict[str, str]:
        return {
            "annualized_return": _format_pct(metrics["annualized_return"]),
            "annualized_volatility": _format_pct(metrics["annualized_volatility"]),
            "sharpe_ratio": _format_num(metrics["sharpe_ratio"]),
            "max_drawdown": _format_pct(metrics["max_drawdown"]),
            "average_daily_turnover": _format_num(metrics["average_daily_turnover"]),
        }

    factor_rows = []
    for row in factor_stats.itertuples(index=False):
        factor_rows.append(
            {
                "factor": row.factor,
                "sharpe_ratio": _format_num(row.sharpe_ratio),
                "annualized_return": _format_pct(row.annualized_return),
                "average_daily_turnover": _format_num(row.average_daily_turnover),
            }
        )

    return template.render(
        full=metric_row(full_metrics),
        insample=metric_row(is_metrics),
        oos=metric_row(oos_metrics),
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
        factor_chart=factor_chart,
        factor_rows=factor_rows,
        clean_dir=str(config.CLEAN_DIR),
    )


def _decode_image(encoded_image: str):
    image_buffer = io.BytesIO(base64.b64decode(encoded_image))
    return plt.imread(image_buffer, format="png")


def _write_pdf_fallback(
    output_path: Path,
    full_metrics: dict[str, float],
    is_metrics: dict[str, float],
    oos_metrics: dict[str, float],
    equity_chart: str,
    drawdown_chart: str,
    factor_chart: str,
    factor_stats: pd.DataFrame,
) -> Path:
    def metric_line(title: str, metrics: dict[str, float]) -> str:
        return (
            f"{title}: Return {_format_pct(metrics['annualized_return'])} | "
            f"Vol {_format_pct(metrics['annualized_volatility'])} | "
            f"Sharpe {_format_num(metrics['sharpe_ratio'])} | "
            f"Max DD {_format_pct(metrics['max_drawdown'])} | "
            f"Turnover {_format_num(metrics['average_daily_turnover'])}"
        )

    equity_img = _decode_image(equity_chart)
    drawdown_img = _decode_image(drawdown_chart)
    factor_img = _decode_image(factor_chart)

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.96, "Statistical Arbitrage Research Report", fontsize=21, fontweight="bold", color="#0b3954")
        fig.text(
            0.06,
            0.935,
            "Market-neutral equity stat arb across S&P 500 constituents, 2016-01 through 2024-12.",
            fontsize=10.5,
            color="#486581",
        )
        fig.text(0.06, 0.895, "Executive Summary & Philosophy", fontsize=14, fontweight="bold", color="#102a43")
        fig.text(
            0.06,
            0.82,
            "This strategy is a market-neutral statistical arbitrage process that seeks to capture transient\n"
            "mispricings in large-cap equities. It blends short-horizon mean reversion, volume dislocations,\n"
            "and medium-horizon momentum, then removes broad market exposure so the portfolio emphasizes\n"
            "stock-specific opportunity instead of outright market direction.",
            fontsize=10.5,
            color="#243b53",
            linespacing=1.5,
        )
        fig.text(0.06, 0.76, "Data Partitioning (IS vs OOS)", fontsize=14, fontweight="bold", color="#102a43")
        fig.text(
            0.06,
            0.705,
            "In-Sample (2016-01 to 2020-12): used for factor discovery and tuning.\n"
            "Out-of-Sample (2021-01 to 2024-12): blind validation window with no parameter tweaking.",
            fontsize=10.5,
            color="#243b53",
            linespacing=1.5,
        )
        fig.text(0.06, 0.655, "Backtest Assumptions", fontsize=14, fontweight="bold", color="#102a43")
        fig.text(
            0.06,
            0.585,
            "T+1 Execution: signals generated at the close of Day T are executed at the close of Day T+1.\n"
            "Transaction Costs: 5 bps are deducted per unit of turnover.\n"
            "Slippage & Shorting: assumes full borrow availability and negligible impact in large-cap names.",
            fontsize=10.5,
            color="#243b53",
            linespacing=1.5,
        )
        fig.text(0.06, 0.535, "Performance Snapshot", fontsize=14, fontweight="bold", color="#102a43")
        fig.text(0.06, 0.507, metric_line("Full Period", full_metrics), fontsize=10.2, color="#243b53")
        fig.text(0.06, 0.485, metric_line("In-Sample", is_metrics), fontsize=10.2, color="#243b53")
        fig.text(0.06, 0.463, metric_line("Out-of-Sample", oos_metrics), fontsize=10.2, color="#243b53")
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.32])
        ax.imshow(equity_img)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.06, 0.96, "Risk & Attribution", fontsize=19, fontweight="bold", color="#0b3954")
        fig.text(
            0.06,
            0.93,
            "The drawdown view highlights peak-to-trough risk, while standalone factor statistics help\n"
            "separate diversification benefits from any one factor's contribution or turnover burden.",
            fontsize=10.5,
            color="#486581",
            linespacing=1.5,
        )
        ax1 = fig.add_axes([0.06, 0.56, 0.88, 0.25])
        ax1.imshow(drawdown_img)
        ax1.axis("off")
        ax2 = fig.add_axes([0.06, 0.23, 0.88, 0.23])
        ax2.imshow(factor_img)
        ax2.axis("off")
        table_ax = fig.add_axes([0.06, 0.05, 0.88, 0.12])
        table_ax.axis("off")
        cell_text = [
            [
                row.factor,
                _format_num(row.sharpe_ratio),
                _format_pct(row.annualized_return),
                _format_num(row.average_daily_turnover),
            ]
            for row in factor_stats.itertuples(index=False)
        ]
        table = table_ax.table(
            cellText=cell_text,
            colLabels=["Factor", "Standalone Sharpe", "Ann. Return", "Avg. Turnover"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 1.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output_path


def generate_report(output_path: Path | str = config.FINAL_REPORT_FILE) -> Path:
    sns.set_theme(style="whitegrid", context="talk")

    results, ret, vol, benchmark = _load_inputs()
    full_metrics = compute_metrics(results["daily_return"], results["turnover"])
    is_metrics = _compute_period_metrics(results, end=config.IN_SAMPLE_END)
    oos_metrics = _compute_period_metrics(results, start=config.OUT_OF_SAMPLE_START)

    equity_chart = _plot_equity_curve(results)
    drawdown_chart = _plot_drawdown(results)
    factor_stats = _compute_factor_attribution(ret, vol, benchmark)
    factor_chart = _plot_factor_attribution(factor_stats)

    html_string = _build_html(
        full_metrics=full_metrics,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
        factor_chart=factor_chart,
        factor_stats=factor_stats,
    )

    output_path = Path(output_path)
    if WEASYPRINT_AVAILABLE:
        HTML(string=html_string, base_url=str(PROJECT_ROOT)).write_pdf(output_path)
    else:
        _write_pdf_fallback(
            output_path=output_path,
            full_metrics=full_metrics,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            equity_chart=equity_chart,
            drawdown_chart=drawdown_chart,
            factor_chart=factor_chart,
            factor_stats=factor_stats,
        )
    return output_path
