#!/usr/bin/env python3
"""
comparar curvas de aprendizaje de CartPole entre la implementacion custom y SB3 PPO.

el script espera ``cartpole_custom_rewards.npy`` y
``cartpole_sb3_rewards.npy`` (cada uno almacenando rewards por episodio) y produce:
    - ``cartpole_comparison.png``: curvas de media movil (ventana configurable) con
      la linea de solucion en 475, plus a small table of summary statistics.
    - salida de consola mostrando tiempo para resolver, media/std final y cualquier archivo faltante.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
CUSTOM_REWARDS = BASE_DIR / "cartpole_custom_rewards.npy"
SB3_REWARDS = BASE_DIR / "cartpole_sb3_rewards.npy"
OUTPUT_PATH = BASE_DIR / "cartpole_comparison.png"
DEFAULT_ROLLING_WINDOW = 100
SOLVE_THRESHOLD = 475.0


def load_rewards(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    if data.ndim != 1:
        raise ValueError(f"{path} must contain a 1-D array of episode rewards.")
    return data


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def time_to_solve(values: np.ndarray, window: int, threshold: float) -> int | None:
    if len(values) < window:
        return None
    smoothed = rolling_mean(values, window)
    for idx in range(window - 1, len(smoothed)):
        if smoothed[idx] >= threshold:
            return idx + 1  
    return None


def build_stats(name: str, rewards: np.ndarray, window: int) -> dict:
    final_window = rewards[-window:] if len(rewards) >= window else rewards
    mean_key = f"mean_last_{window}"
    std_key = f"std_last_{window}"
    return {
        "implementation": name,
        "episodes": len(rewards),
        "time_to_solve": time_to_solve(rewards, window, SOLVE_THRESHOLD),
        mean_key: float(np.mean(final_window)) if len(final_window) else float("nan"),
        std_key: float(np.std(final_window)) if len(final_window) else float("nan"),
    }


def plot_comparison(
    custom_rewards: np.ndarray,
    sb3_rewards: np.ndarray,
    output_path: Path,
    window: int,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    custom_curve = rolling_mean(custom_rewards, window)
    sb3_curve = rolling_mean(sb3_rewards, window)
    x_custom = np.arange(1, len(custom_curve) + 1)
    x_sb3 = np.arange(1, len(sb3_curve) + 1)

    plt.plot(x_custom, custom_curve, label="Custom PPO", linewidth=2)
    plt.plot(x_sb3, sb3_curve, label="SB3 PPO", linewidth=2)
    plt.axhline(SOLVE_THRESHOLD, color="red", linestyle="--", label="Solve threshold (475)")
    plt.xlabel("Episodio")
    plt.ylabel(f"Reward (media móvil {window})")
    plt.title("CartPole-v1: Custom PPO vs Stable Baselines 3")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CartPole PPO runs.")
    parser.add_argument(
        "--custom",
        type=Path,
        default=CUSTOM_REWARDS,
        help="Path to cartpole_custom_rewards.npy (default: %(default)s)",
    )
    parser.add_argument(
        "--sb3",
        type=Path,
        default=SB3_REWARDS,
        help="Path to cartpole_sb3_rewards.npy (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for comparison figure (default: %(default)s)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help="Rolling window size for smoothing and stats (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    window = max(1, args.window)

    custom_rewards = load_rewards(args.custom)
    sb3_rewards = load_rewards(args.sb3)

    plot_comparison(custom_rewards, sb3_rewards, args.output, window)

    stats = [
        build_stats("Custom PPO", custom_rewards, window),
        build_stats("SB3 PPO", sb3_rewards, window),
    ]
    stats_df = pd.DataFrame(stats)

    print("\nCARTPOLE – COMPARACIÓN DE IMPLEMENTACIONES")
    print(stats_df.to_string(index=False)) 
    print(f"Ventana de media móvil: {window} episodios")
    print(f"\nFigura guardada en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

