import pandas as pd
from engine import compute_atr


def _smma(series: pd.Series, period: int) -> pd.Series:
    result = series.copy() * float("nan")
    sma_start = series.iloc[:period].mean()
    result.iloc[period - 1] = sma_start
    for i in range(period, len(series)):
        result.iloc[i] = (result.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
    return result


class SmmaCrossover:
    def __init__(self, config: dict):
        cfg = config["strategies"]["smma_crossover"]
        self.fast = cfg["fast_period"]
        self.slow = cfg["slow_period"]
        self.atr_period = cfg["atr_period"]
        self.atr_mult = cfg["atr_multiplier"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = _smma(df["close"], self.fast)
        slow = _smma(df["close"], self.slow)
        atr = compute_atr(df, self.atr_period)

        signals = pd.DataFrame(index=df.index)
        signals["entry"] = 0
        signals["exit"] = False
        signals["stop_loss"] = 0.0

        in_trade = False
        direction = 0

        for i in range(1, len(df)):
            prev_fast = fast.iloc[i - 1]
            prev_slow = slow.iloc[i - 1]
            curr_fast = fast.iloc[i]
            curr_slow = slow.iloc[i]
            price = df["close"].iloc[i]
            curr_atr = atr.iloc[i]

            if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow) or pd.isna(curr_atr):
                continue

            bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
            bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

            if not in_trade and bullish_cross:
                signals.at[df.index[i], "entry"] = 1
                signals.at[df.index[i], "stop_loss"] = price - self.atr_mult * curr_atr
                in_trade = True
                direction = 1
            elif in_trade and direction == 1 and bearish_cross:
                signals.at[df.index[i], "exit"] = True
                in_trade = False
                direction = 0

        return signals
