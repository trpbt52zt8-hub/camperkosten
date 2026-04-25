import pandas as pd
from engine import compute_atr


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


class RsiMeanReversion:
    def __init__(self, config: dict):
        cfg = config["strategies"]["rsi_mean_reversion"]
        self.rsi_period = cfg["rsi_period"]
        self.oversold = cfg["oversold"]
        self.overbought = cfg["overbought"]
        self.exit_level = cfg["exit_level"]
        self.atr_period = cfg["atr_period"]
        self.atr_mult = cfg["atr_multiplier"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = _rsi(df["close"], self.rsi_period)
        atr = compute_atr(df, self.atr_period)

        signals = pd.DataFrame(index=df.index)
        signals["entry"] = 0
        signals["exit"] = False
        signals["stop_loss"] = 0.0

        in_trade = False
        direction = 0

        for i in range(1, len(df)):
            curr_rsi = rsi.iloc[i]
            price = df["close"].iloc[i]
            curr_atr = atr.iloc[i]

            if pd.isna(curr_rsi) or pd.isna(curr_atr):
                continue

            if not in_trade:
                if curr_rsi < self.oversold:
                    signals.at[df.index[i], "entry"] = 1
                    signals.at[df.index[i], "stop_loss"] = price - self.atr_mult * curr_atr
                    in_trade = True
                    direction = 1
                elif curr_rsi > self.overbought:
                    signals.at[df.index[i], "entry"] = -1
                    signals.at[df.index[i], "stop_loss"] = price + self.atr_mult * curr_atr
                    in_trade = True
                    direction = -1
            else:
                if direction == 1 and curr_rsi > self.exit_level:
                    signals.at[df.index[i], "exit"] = True
                    in_trade = False
                    direction = 0
                elif direction == -1 and curr_rsi < self.exit_level:
                    signals.at[df.index[i], "exit"] = True
                    in_trade = False
                    direction = 0

        return signals
