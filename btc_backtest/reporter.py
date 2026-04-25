import os
import webbrowser
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Template
from engine import BacktestResult


REPORT_DIR = "output"


def _ensure_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)


def export_trade_logs(results: list[BacktestResult]):
    _ensure_dir()
    for r in results:
        if not r.trades:
            continue
        rows = []
        for t in r.trades:
            rows.append({
                "entry_time": t.entry_time,
                "entry_price": round(t.entry_price, 2),
                "direction": "long" if t.direction == 1 else "short",
                "exit_time": t.exit_time,
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "size": round(t.size, 6),
                "pnl": round(t.pnl, 2) if t.pnl else None,
                "pnl_pct": round(t.pnl_pct, 3) if t.pnl_pct else None,
                "r_multiple": round(t.r_multiple, 3) if t.r_multiple else None,
                "exit_reason": t.exit_reason
            })
        df = pd.DataFrame(rows)
        path = os.path.join(REPORT_DIR, f"{r.strategy_name}_trades.csv")
        df.to_csv(path, index=False)
        print(f"Trade log saved: {path}")


def build_comparison_table(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r.metrics
        rows.append({
            "Strategy": r.strategy_name,
            "Trades": m["total_trades"],
            "Win Rate %": f"{m['win_rate']:.1f}",
            "Avg R": f"{m['avg_r']:.2f}",
            "Max DD %": f"{m['max_drawdown_pct']:.1f}",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Return %": f"{m['total_return_pct']:.1f}"
        })
    return pd.DataFrame(rows)


def plot_equity_curves(results: list[BacktestResult]) -> str:
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for r, color in zip(results, colors):
        ax.plot(r.equity_curve.index, r.equity_curve.values, label=r.strategy_name, color=color, linewidth=1.5)

    ax.set_title("Equity Curves — BTCUSDT H1", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = os.path.join(REPORT_DIR, "equity_curves.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Equity curve saved: {path}")
    return path


def generate_html_report(results: list[BacktestResult], table: pd.DataFrame, chart_path: str) -> str:
    _ensure_dir()

    template_str = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<title>BTC Backtest Report</title>
<style>
  body { font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: 40px auto; background: #f8f9fa; color: #212529; }
  h1 { color: #1a1a2e; border-bottom: 3px solid #2196F3; padding-bottom: 8px; }
  h2 { color: #16213e; margin-top: 40px; }
  table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  th { background: #2196F3; color: white; padding: 12px 16px; text-align: left; font-size: 0.9rem; }
  td { padding: 10px 16px; border-bottom: 1px solid #e9ecef; font-size: 0.9rem; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) { background: #f1f8ff; }
  .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }
  .metric-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #2196F3; }
  .metric-card h3 { margin: 0 0 12px; font-size: 1rem; color: #495057; }
  .metric-card .value { font-size: 1.6rem; font-weight: bold; color: #1a1a2e; }
  .metric-card .label { font-size: 0.8rem; color: #6c757d; margin-top: 4px; }
  .positive { color: #28a745 !important; }
  .negative { color: #dc3545 !important; }
  img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .footer { margin-top: 40px; color: #6c757d; font-size: 0.85rem; border-top: 1px solid #dee2e6; padding-top: 16px; }
  .strategy-section { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
  .badge-long { background: #d4edda; color: #155724; }
  .badge-short { background: #f8d7da; color: #721c24; }
  trade-table td, .trade-table th { padding: 7px 12px; font-size: 0.82rem; }
</style>
</head>
<body>
<h1>BTC Backtest Report</h1>
<p>Symbol: <strong>BTCUSDT H1</strong> &nbsp;|&nbsp; Period: <strong>{{ period }}</strong> &nbsp;|&nbsp; Initial capital: <strong>${{ initial_capital }}</strong> &nbsp;|&nbsp; Fee: <strong>0.1% per trade</strong></p>

<h2>Vergelijkingstabel</h2>
<table>
  <thead><tr>{% for col in table_cols %}<th>{{ col }}</th>{% endfor %}</tr></thead>
  <tbody>
  {% for row in table_rows %}
  <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
  {% endfor %}
  </tbody>
</table>

<h2>Equity Curves</h2>
<img src="equity_curves.png" alt="Equity Curves">

{% for result in results %}
<div class="strategy-section">
  <h2>{{ result.strategy_name }}</h2>
  <div class="metric-grid">
    <div class="metric-card">
      <h3>Trades</h3>
      <div class="value">{{ result.metrics.total_trades }}</div>
    </div>
    <div class="metric-card">
      <h3>Win Rate</h3>
      <div class="value {% if result.metrics.win_rate >= 50 %}positive{% else %}negative{% endif %}">{{ "%.1f"|format(result.metrics.win_rate) }}%</div>
    </div>
    <div class="metric-card">
      <h3>Avg R-Multiple</h3>
      <div class="value {% if result.metrics.avg_r >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(result.metrics.avg_r) }}R</div>
    </div>
    <div class="metric-card">
      <h3>Max Drawdown</h3>
      <div class="value negative">{{ "%.1f"|format(result.metrics.max_drawdown_pct) }}%</div>
    </div>
    <div class="metric-card">
      <h3>Sharpe Ratio</h3>
      <div class="value {% if result.metrics.sharpe >= 1 %}positive{% elif result.metrics.sharpe >= 0 %}{% else %}negative{% endif %}">{{ "%.2f"|format(result.metrics.sharpe) }}</div>
    </div>
    <div class="metric-card">
      <h3>Total Return</h3>
      <div class="value {% if result.metrics.total_return_pct >= 0 %}positive{% else %}negative{% endif %}">{{ "%.1f"|format(result.metrics.total_return_pct) }}%</div>
    </div>
  </div>

  {% if result.trade_rows %}
  <h3>Trade Log (eerste 30 trades)</h3>
  <table class="trade-table">
    <thead><tr>
      <th>Entry</th><th>Entry Price</th><th>Direction</th>
      <th>Exit</th><th>Exit Price</th><th>PnL $</th><th>PnL %</th><th>R</th><th>Reden</th>
    </tr></thead>
    <tbody>
    {% for t in result.trade_rows %}
    <tr>
      <td>{{ t.entry_time }}</td>
      <td>${{ t.entry_price }}</td>
      <td><span class="badge {% if t.direction == 'long' %}badge-long{% else %}badge-short{% endif %}">{{ t.direction }}</span></td>
      <td>{{ t.exit_time }}</td>
      <td>${{ t.exit_price }}</td>
      <td class="{% if t.pnl >= 0 %}positive{% else %}negative{% endif %}">${{ t.pnl }}</td>
      <td class="{% if t.pnl_pct >= 0 %}positive{% else %}negative{% endif %}">{{ t.pnl_pct }}%</td>
      <td class="{% if t.r_multiple >= 0 %}positive{% else %}negative{% endif %}">{{ t.r_multiple }}R</td>
      <td>{{ t.exit_reason }}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
</div>
{% endfor %}

<div class="footer">
  Gegenereerd op {{ generated_at }} &nbsp;|&nbsp; BTC Backtest Engine v1.0 &nbsp;|&nbsp; Geen live trading — uitsluitend historische simulatie
</div>
</body>
</html>"""

    chart_rel = os.path.basename(chart_path)
    equity_start = results[0].equity_curve.index[0].strftime("%Y-%m-%d")
    equity_end = results[0].equity_curve.index[-1].strftime("%Y-%m-%d")

    results_ctx = []
    for r in results:
        trade_rows = []
        for t in r.trades[:30]:
            trade_rows.append({
                "entry_time": str(t.entry_time)[:16],
                "entry_price": round(t.entry_price, 2),
                "direction": "long" if t.direction == 1 else "short",
                "exit_time": str(t.exit_time)[:16] if t.exit_time else "",
                "exit_price": round(t.exit_price, 2) if t.exit_price else 0,
                "pnl": round(t.pnl, 2) if t.pnl else 0,
                "pnl_pct": round(t.pnl_pct, 3) if t.pnl_pct else 0,
                "r_multiple": round(t.r_multiple, 3) if t.r_multiple else 0,
                "exit_reason": t.exit_reason
            })
        results_ctx.append({"strategy_name": r.strategy_name, "metrics": r.metrics, "trade_rows": trade_rows})

    html = Template(template_str).render(
        period=f"{equity_start} – {equity_end}",
        initial_capital="10,000",
        table_cols=list(table.columns),
        table_rows=table.values.tolist(),
        results=results_ctx,
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        chart_path=chart_rel
    )

    report_path = os.path.join(REPORT_DIR, "report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"HTML report saved: {report_path}")
    return report_path


def print_comparison_table(table: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  BTC BACKTEST — VERGELIJKINGSTABEL")
    print("=" * 65)
    col_w = [20, 7, 10, 7, 10, 8, 10]
    header = "".join(str(c).ljust(w) for c, w in zip(table.columns, col_w))
    print(header)
    print("-" * 65)
    for _, row in table.iterrows():
        print("".join(str(v).ljust(w) for v, w in zip(row.values, col_w)))
    print("=" * 65 + "\n")


def generate_report(results: list[BacktestResult], open_browser: bool = True) -> str:
    export_trade_logs(results)
    table = build_comparison_table(results)
    print_comparison_table(table)
    chart_path = plot_equity_curves(results)
    report_path = generate_html_report(results, table, chart_path)

    if open_browser:
        abs_path = os.path.abspath(report_path)
        webbrowser.open(f"file://{abs_path}")
        print(f"\nRapport geopend in browser: {abs_path}")

    return report_path
