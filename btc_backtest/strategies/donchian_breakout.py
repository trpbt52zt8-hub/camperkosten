import pandas as pd
from engine import compute_atr


class DonchianBreakout:
    def __init__(self, config: dict):
        cfg = config["strategies"]["donchian_breakout"]
        self.period = cfg["channel_period"]
        self.atr_period = cfg["atr_period"]
        self.atr_mult = cfg["atr_multiplier"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        high_channel = df["high"].shift(1).rolling(self.period).max()
        low_channel = df["low"].shift(1).rolling(self.period).min()
        atr = compute_atr(df, self.atr_period)

        signals = pd.DataFrame(index=df.index)
        signals["entry"] = 0
        signals["exit"] = False
        signals["stop_loss"] = 0.0

        in_trade = False
        direction = 0

        for i in range(self.period, len(df)):
            price = df["close"].iloc[i]
            curr_high = high_channel.iloc[i]
            curr_low = low_channel.iloc[i]
            curr_atr = atr.iloc[i]

            if pd.isna(curr_high) or pd.isna(curr_low) or pd.isna(curr_atr):
                continue

            if not in_trade:
                if price > curr_high:
                    signals.at[df.index[i], "entry"] = 1
                    signals.at[df.index[i], "stop_loss"] = price - self.atr_mult * curr_atr
                    in_trade = True
                    direction = 1
                elif price < curr_low:
                    signals.at[df.index[i], "entry"] = -1
                    signals.at[df.index[i], "stop_loss"] = price + self.atr_mult * curr_atr
                    in_trade = True
                    direction = -1
            else:
                if direction == 1 and price < curr_low:
                    signals.at[df.index[i], "exit"] = True
                    in_trade = False
                    direction = 0
                elif direction == -1 and price > curr_high:
                    signals.at[df.index[i], "exit"] = True
                    in_trade = False
                    direction = 0

        return signals
