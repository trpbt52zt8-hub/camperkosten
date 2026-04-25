import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "strategies"))

from data_fetcher import load_config, get_data
from engine import BacktestEngine
from strategies.smma_crossover import SmmaCrossover
from strategies.rsi_mean_reversion import RsiMeanReversion
from strategies.donchian_breakout import DonchianBreakout
from reporter import generate_report


def main():
    demo_mode = "--demo" in sys.argv

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(config_path)

    os.chdir(os.path.dirname(__file__))

    df = get_data(config, demo=demo_mode)

    engine = BacktestEngine(config)

    strategies = [
        ("SMMA Crossover", SmmaCrossover(config)),
        ("RSI Mean Reversion", RsiMeanReversion(config)),
        ("Donchian Breakout", DonchianBreakout(config)),
    ]

    results = []
    for name, strategy in strategies:
        print(f"\nRunning: {name}...")
        result = engine.run(df, strategy, name)
        results.append(result)
        m = result.metrics
        print(f"  Trades: {m['total_trades']} | Win: {m['win_rate']:.1f}% | Return: {m['total_return_pct']:.1f}%")

    generate_report(results)


if __name__ == "__main__":
    main()
