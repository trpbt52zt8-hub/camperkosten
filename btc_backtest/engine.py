import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1 = long, -1 = short
    stop_loss: float
    size: float  # units of BTC
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    exit_reason: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    metrics: dict = field(default_factory=dict)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_metrics(trades: list, equity_curve: pd.Series, initial_capital: float) -> dict:
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "avg_r": 0,
            "max_drawdown_pct": 0, "sharpe": 0, "total_return_pct": 0
        }

    pnls = [t.pnl for t in trades if t.pnl is not None]
    r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
    wins = [p for p in pnls if p > 0]

    # Max drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min() * 100

    # Sharpe (annualised, assuming hourly data)
    returns = equity_curve.pct_change().dropna()
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)

    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital * 100

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(pnls) * 100 if pnls else 0,
        "avg_r": np.mean(r_multiples) if r_multiples else 0,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "total_return_pct": total_return
    }


class BacktestEngine:
    def __init__(self, config: dict):
        self.initial_capital = config["backtest"]["initial_capital"]
        self.risk_per_trade = config["backtest"]["risk_per_trade"]
        self.fee_rate = config["backtest"]["fee_rate"]

    def _position_size(self, capital: float, entry: float, stop: float) -> float:
        risk_amount = capital * self.risk_per_trade
        distance = abs(entry - stop)
        if distance == 0:
            return 0
        return risk_amount / distance

    def _apply_fee(self, capital: float, size: float, price: float) -> float:
        return capital - size * price * self.fee_rate

    def run(self, df: pd.DataFrame, strategy, name: str) -> BacktestResult:
        signals = strategy.generate_signals(df)
        capital = self.initial_capital
        equity = []
        trades = []
        open_trade: Optional[Trade] = None

        for i, (ts, row) in enumerate(df.iterrows()):
            sig = signals.iloc[i]
            price = row["close"]

            # Check stop loss on open trade
            if open_trade is not None:
                hit_stop = False
                if open_trade.direction == 1 and row["low"] <= open_trade.stop_loss:
                    exit_price = open_trade.stop_loss
                    hit_stop = True
                elif open_trade.direction == -1 and row["high"] >= open_trade.stop_loss:
                    exit_price = open_trade.stop_loss
                    hit_stop = True

                if hit_stop:
                    open_trade = self._close_trade(open_trade, ts, exit_price, "stop_loss", capital)
                    capital += open_trade.pnl
                    capital = self._apply_fee(capital, open_trade.size, exit_price)
                    trades.append(open_trade)
                    open_trade = None

            # Check exit signal on open trade
            if open_trade is not None and sig.get("exit", False):
                open_trade = self._close_trade(open_trade, ts, price, "signal", capital)
                capital += open_trade.pnl
                capital = self._apply_fee(capital, open_trade.size, price)
                trades.append(open_trade)
                open_trade = None

            # Open new trade
            if open_trade is None and sig.get("entry") in (1, -1):
                direction = sig["entry"]
                stop = sig.get("stop_loss", 0)
                size = self._position_size(capital, price, stop)
                if size > 0:
                    capital = self._apply_fee(capital, size, price)
                    open_trade = Trade(
                        entry_time=ts,
                        entry_price=price,
                        direction=direction,
                        stop_loss=stop,
                        size=size
                    )

            equity.append(capital if open_trade is None else capital + open_trade.size * price * open_trade.direction)

        # Close any remaining trade at last price
        if open_trade is not None:
            last_price = df["close"].iloc[-1]
            open_trade = self._close_trade(open_trade, df.index[-1], last_price, "end_of_data", capital)
            capital += open_trade.pnl
            trades.append(open_trade)
            equity[-1] = capital

        equity_series = pd.Series(equity, index=df.index)
        metrics = compute_metrics(trades, equity_series, self.initial_capital)

        return BacktestResult(
            strategy_name=name,
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics
        )

    def _close_trade(self, trade: Trade, ts, price: float, reason: str, capital: float) -> Trade:
        pnl = (price - trade.entry_price) * trade.direction * trade.size
        risk = abs(trade.entry_price - trade.stop_loss) * trade.size
        trade.exit_time = ts
        trade.exit_price = price
        trade.pnl = pnl
        trade.pnl_pct = pnl / (trade.entry_price * trade.size) * 100
        trade.r_multiple = pnl / risk if risk > 0 else 0
        trade.exit_reason = reason
        return trade
