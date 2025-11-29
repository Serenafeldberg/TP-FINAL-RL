#!/usr/bin/env python3
"""
construir tablas LaTeX listas para el informe final.

tablas generadas (guardadas bajo ``report_tables/`` por defecto):
    - ``configurations.tex``    : hiperparametros por configuracion Flappy
    - ``flappy_results.tex``    : metricas de ranking (mean/std/max) para Flappy runs
    - ``cartpole_comparison.tex`` : Custom PPO vs SB3 statistics on CartPole
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import analyze_flappy_results as analyzer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SAVED_MODELS_DIR = BASE_DIR / "savedModels"
DEFAULT_TABLE_DIR = BASE_DIR / "report_tables"
DEFAULT_CARTPOLE_CUSTOM = BASE_DIR / "cartpole_custom_rewards.npy"
DEFAULT_CARTPOLE_SB3 = BASE_DIR / "cartpole_sb3_rewards.npy"
ROLLING_WINDOW = 100


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for the RL final report."
    )
    parser.add_argument(
        "--saved-models-dir",
        type=Path,
        default=DEFAULT_SAVED_MODELS_DIR,
        help="Directory with config_* folders (default: %(default)s)",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help="Output directory for LaTeX tables (default: %(default)s)",
    )
    parser.add_argument(
        "--cartpole-custom",
        type=Path,
        default=DEFAULT_CARTPOLE_CUSTOM,
        help="Path to cartpole_custom_rewards.npy (default: %(default)s)",
    )
    parser.add_argument(
        "--cartpole-sb3",
        type=Path,
        default=DEFAULT_CARTPOLE_SB3,
        help="Path to cartpole_sb3_rewards.npy (default: %(default)s)",
    )
    parser.add_argument(
        "--max-config-id",
        type=int,
        default=18,
        help="Highest Flappy config id to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Rows to include in the Flappy results table (default: %(default)s)",
    )
    return parser.parse_args(argv)


def load_config_metadata(config_dir: Path) -> Optional[dict]:
    config_path = config_dir / "config.json"
    if not config_path.exists():
        return None
    data = pd.read_json(config_path)
    if data.empty:
        return None
    return data.to_dict()


def build_config_table(ranking_df: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "config_id",
        "config_name",
        "learning_rate",
        "entropy_coef",
        "n_epochs",
        "hidden_size",
        "batch_size",
    ]
    configs_df = ranking_df.loc[:, columns].copy()
    configs_df.rename(
        columns={
            "config_id": "ID",
            "config_name": "Nombre",
            "learning_rate": "LR",
            "entropy_coef": "Entropy",
            "n_epochs": "Epochs",
            "hidden_size": "Hidden",
            "batch_size": "Batch",
        },
        inplace=True,
    )
    latex = configs_df.to_latex(
        index=False,
        float_format="%.2g",
        column_format="l" * len(configs_df.columns),
    )
    output_path.write_text(latex)


def build_flappy_results_table(
    ranking_df: pd.DataFrame, output_path: Path, top_k: int
) -> None:
    columns = [
        "rank",
        "config_id",
        "config_name",
        "mean_reward_last",
        "std_reward_last",
        "max_reward_last",
        "total_episodes",
    ]
    df = ranking_df.loc[:, columns].head(top_k).copy()
    df.rename(
        columns={
            "rank": "Rank",
            "config_id": "ID",
            "config_name": "Nombre",
            "mean_reward_last": "Media",
            "std_reward_last": "Std",
            "max_reward_last": "Máx",
            "total_episodes": "Episodios",
        },
        inplace=True,
    )
    latex = df.to_latex(
        index=False,
        float_format="%.2f",
        column_format="l" * len(df.columns),
    )
    output_path.write_text(latex)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def time_to_solve(values: np.ndarray, window: int, threshold: float) -> Optional[int]:
    smoothed = rolling_mean(values, window)
    for idx in range(window - 1, smoothed.size):
        if smoothed[idx] >= threshold:
            return idx + 1
    return None


def build_cartpole_table(
    custom_rewards_path: Path,
    sb3_rewards_path: Path,
    output_path: Path,
) -> None:
    custom = np.load(custom_rewards_path)
    sb3 = np.load(sb3_rewards_path)
    entries = []
    for label, rewards, threshold in [
        ("Custom PPO", custom, 475.0),
        ("SB3 PPO", sb3, 475.0),
    ]:
        final_window = rewards[-ROLLING_WINDOW:]
        mean_final = float(np.mean(final_window))
        std_final = float(np.std(final_window))
        solve_ep = time_to_solve(rewards, ROLLING_WINDOW, threshold)
        entries.append(
            {
                "Implementación": label,
                "Episodios": len(rewards),
                "Mean$_{100}$": mean_final,
                "Std$_{100}$": std_final,
                "Episodio Resuelto": solve_ep if solve_ep else "—",
            }
        )
    df = pd.DataFrame(entries)
    latex = df.to_latex(
        index=False,
        float_format="%.2f",
        column_format="lcccc",
    )
    output_path.write_text(latex)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    saved_models_dir = args.saved_models_dir.expanduser().resolve()
    table_dir = args.table_dir.expanduser().resolve()
    ensure_dir(table_dir)

    ranking_df, missing = analyzer.analyze_configs(
        saved_models_dir=saved_models_dir,
        max_config_id=args.max_config_id,
        rolling_window=ROLLING_WINDOW,
    )
    if ranking_df.empty:
        print("[ERROR] No se pudieron generar estadísticas de configuraciones.")
        for issue in missing:
            print(f"  - {issue}")
        return 1
    if missing:
        print("Advertencias durante el análisis:")
        for issue in missing:
            print(f"  - {issue}")

    build_config_table(ranking_df, table_dir / "configurations.tex")
    build_flappy_results_table(
        ranking_df,
        table_dir / "flappy_results.tex",
        args.top_k,
    )
    build_cartpole_table(
        args.cartpole_custom.expanduser().resolve(),
        args.cartpole_sb3.expanduser().resolve(),
        table_dir / "cartpole_comparison.tex",
    )

    print("Tablas LaTeX guardadas en:", table_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

