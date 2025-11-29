#!/usr/bin/env python3
"""
generar todas las figuras publicables requeridas para el informe final de RL.

salidas (todas guardadas bajo ``report_figures/`` por defecto, 300 dpi):
    - ``architecture.png``              : diagrama Actor-Critic
    - ``learning_curves_top5.png``      : curvas de recompensa para las 5 mejores configuraciones Flappy
    - ``baseline_vs_best.png``          : baseline vs las 3 mejores configuraciones
    - ``hyperparams_ablation.png``      : graficos de barras comparando los hiperparametros clave
    - ``entropy_exploration.png``       : evolucion de la entropia para las 3 mejores configuraciones
    - ``cartpole_comparison.png``       : Custom PPO vs SB3 en CartPole
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# agregar scripts/analysis/ al path para importar analyze_flappy_results
project_root = Path(__file__).parent.parent.parent
analysis_path = project_root / "scripts" / "analysis"
if str(analysis_path) not in sys.path:
    sys.path.insert(0, str(analysis_path))

import analyze_flappy_results as analyzer

BASE_DIR = project_root  # TP-FINAL-RL/
DEFAULT_SAVED_MODELS_DIR = BASE_DIR / "savedModels"
DEFAULT_FIG_DIR = BASE_DIR / "report_figures"
DEFAULT_CARTPOLE_CUSTOM = BASE_DIR / "data" / "probe_envs" / "cartpole_custom_rewards.npy"
DEFAULT_CARTPOLE_SB3 = BASE_DIR / "data" / "probe_envs" / "cartpole_sb3_rewards.npy"
ROLLING_WINDOW = 100
ENTROPY_WINDOW = 200
SOLVE_THRESHOLD_CARTPOLE = 475.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all figures required for the RL final report."
    )
    parser.add_argument(
        "--saved-models-dir",
        type=Path,
        default=DEFAULT_SAVED_MODELS_DIR,
        help="Directory containing config_* folders (default: %(default)s)",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help="Output directory for the figures (default: %(default)s)",
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
        help="Maximum configuration id to scan (default: %(default)s)",
    )
    return parser.parse_args(argv)


def load_rewards(config_dir: Path) -> pd.DataFrame:
    rewards_path = config_dir / "logs" / "rewards.csv"
    df = pd.read_csv(rewards_path)
    return df.sort_values("episode").reset_index(drop=True)


def load_losses(config_dir: Path) -> pd.DataFrame:
    losses_path = config_dir / "logs" / "losses.csv"
    df = pd.read_csv(losses_path)
    return df.sort_values("timestep").reset_index(drop=True)


def get_label(row: pd.Series) -> str:
    return f"Config {int(row['config_id'])} – {row['config_name']}"


def collect_series(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    config_ids: Iterable[int],
    window: int,
) -> List[Tuple[pd.Series, pd.Series, str]]:
    series = []
    for config_id in config_ids:
        config_dir = analyzer.discover_config_dir(saved_models_dir, config_id)
        if config_dir is None:
            continue
        rewards_df = load_rewards(config_dir)
        if rewards_df.empty:
            continue
        smoothed = rewards_df["reward"].rolling(window=window, min_periods=1).mean()
        label = get_label(
            ranking_df.loc[ranking_df["config_id"] == config_id].iloc[0]
        )
        series.append((rewards_df["episode"], smoothed, label))
    return series


def figure_learning_curves_top5(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    top_ids = ranking_df.head(5)["config_id"].tolist()
    data = collect_series(ranking_df, saved_models_dir, top_ids, ROLLING_WINDOW)
    for episodes, rewards, label in data:
        plt.plot(episodes, rewards, linewidth=2, label=label)
    plt.xlabel("Episodio")
    plt.ylabel(f"Reward (media móvil {ROLLING_WINDOW})")
    plt.title("Flappy Bird · Mejores 5 configuraciones (PPO)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_baseline_vs_best(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    best_ids = ranking_df.head(3)["config_id"].tolist()
    config_ids = [1] + [cfg for cfg in best_ids if cfg != 1]
    data = collect_series(ranking_df, saved_models_dir, config_ids, ROLLING_WINDOW)
    for episodes, rewards, label in data:
        plt.plot(episodes, rewards, linewidth=2, label=label)
    plt.xlabel("Episodio")
    plt.ylabel(f"Reward (media móvil {ROLLING_WINDOW})")
    plt.title("Flappy Bird · Baseline vs Top 3")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_entropy_exploration(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    top_ids = ranking_df.head(3)["config_id"].tolist()
    for config_id in top_ids:
        config_dir = analyzer.discover_config_dir(saved_models_dir, config_id)
        if config_dir is None:
            continue
        losses_df = load_losses(config_dir)
        if losses_df.empty:
            continue
        smoothed = (-losses_df["entropy_loss"]).rolling(
            window=ENTROPY_WINDOW, min_periods=1
        ).mean()
        label = get_label(
            ranking_df.loc[ranking_df["config_id"] == config_id].iloc[0]
        )
        plt.plot(losses_df["timestep"], smoothed, linewidth=2, label=label)
    plt.xlabel("Timestep")
    plt.ylabel(f"Entropía (media móvil {ENTROPY_WINDOW})")
    plt.title("Flappy Bird · Evolución de entropía (Top 3)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def aggregate_by_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column)["mean_reward_last"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.rename(
        columns={"mean": "mean_reward", "std": "std_reward", "count": "configs"},
        inplace=True,
    )
    return grouped


def figure_hyperparams_ablation(
    ranking_df: pd.DataFrame,
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    info = [
        ("learning_rate", "Learning Rate"),
        ("hidden_size", "Hidden Size"),
        ("n_epochs", "Epochs"),
        ("batch_size", "Batch Size"),
    ]
    palette = sns.color_palette("viridis", 4)
    for ax, (column, title), color in zip(axes, info, palette):
        grouped = aggregate_by_column(ranking_df, column)
        if grouped.empty:
            ax.set_visible(False)
            continue
        ax.bar(
            grouped[column].astype(str),
            grouped["mean_reward"],
            color=color,
            alpha=0.85,
        )
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Mean Reward (últimos 100)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    fig.suptitle("Flappy Bird · Ablación de hiperparámetros", fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def figure_cartpole_comparison(
    custom_rewards_path: Path,
    sb3_rewards_path: Path,
    fig_path: Path,
) -> None:
    custom = np.load(custom_rewards_path)
    sb3 = np.load(sb3_rewards_path)
    custom_curve = rolling_mean(custom, ROLLING_WINDOW)
    sb3_curve = rolling_mean(sb3, ROLLING_WINDOW)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.arange(1, custom_curve.size + 1),
        custom_curve,
        linewidth=2,
        label="Custom PPO",
    )
    plt.plot(
        np.arange(1, sb3_curve.size + 1),
        sb3_curve,
        linewidth=2,
        label="SB3 PPO",
    )
    plt.axhline(
        SOLVE_THRESHOLD_CARTPOLE,
        color="red",
        linestyle="--",
        label="Umbral de resolución",
    )
    plt.xlabel("Episodio")
    plt.ylabel(f"Reward (media móvil {ROLLING_WINDOW})")
    plt.title("CartPole-v1 · Comparación Custom PPO vs SB3")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def figure_architecture(fig_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        ("Observación\n(normalizada)", (0.1, 0.4)),
        ("Feature Extractor\n(MLP)", (0.35, 0.4)),
        ("Actor Head\n(π(a|s))", (0.65, 0.6)),
        ("Critic Head\n(V(s))", (0.65, 0.2)),
    ]
    for text, (x, y) in boxes:
        rect = plt.Rectangle(
            (x, y),
            0.2,
            0.2,
            edgecolor="#1f2937",
            facecolor="#cbd5f5",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.1,
            y + 0.1,
            text,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    arrows = [
        ((0.3, 0.5), (0.35, 0.5)),
        ((0.55, 0.5), (0.65, 0.7)),
        ((0.55, 0.5), (0.65, 0.3)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=2, color="#111827"),
        )
    ax.text(
        0.1,
        0.55,
        "Obs / 400\n+ RunningNorm",
        fontsize=11,
        color="#1f2937",
    )
    plt.title("Arquitectura Actor-Critic (PPO)", fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    saved_models_dir = args.saved_models_dir.expanduser().resolve()
    fig_dir = args.fig_dir.expanduser().resolve()
    ensure_dir(fig_dir)
    sns.set_palette("colorblind")

    ranking_df, missing = analyzer.analyze_configs(
        saved_models_dir=saved_models_dir,
        max_config_id=args.max_config_id,
        rolling_window=ROLLING_WINDOW,
    )
    if ranking_df.empty:
        print("[ERROR] No se encontraron configuraciones para graficar.")
        for issue in missing:
            print(f"  - {issue}")
        return 1
    if missing:
        print("Advertencias durante el análisis:")
        for issue in missing:
            print(f"  - {issue}")

    figure_architecture(fig_dir / "architecture.png")
    figure_learning_curves_top5(
        ranking_df, saved_models_dir, fig_dir / "learning_curves_top5.png"
    )
    figure_baseline_vs_best(
        ranking_df, saved_models_dir, fig_dir / "baseline_vs_best.png"
    )
    figure_entropy_exploration(
        ranking_df, saved_models_dir, fig_dir / "entropy_exploration.png"
    )
    figure_hyperparams_ablation(
        ranking_df,
        fig_dir / "hyperparams_ablation.png",
    )
    figure_cartpole_comparison(
        args.cartpole_custom.expanduser().resolve(),
        args.cartpole_sb3.expanduser().resolve(),
        fig_dir / "cartpole_comparison.png",
    )

    print("Figuras generadas en:", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
