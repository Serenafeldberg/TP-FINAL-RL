#!/usr/bin/env python3
"""
generar visualizaciones comparativas para las experimentaciones PPO Flappy Bird.

el script lee cada carpeta ``config_*`` bajo ``savedModels/``, reproduce el
ranking used in ``analyze_flappy_results.py`` (mean reward over the last
``rolling_window`` episodios), y genera figuras PNG para curvas de entrenamiento y
resultados de evaluacion.

Analisis de Entrenamiento:
1. ``top5_learning_curves.png``       – Rewards (rolling mean=100) para las 5 configuraciones.
2. ``baseline_vs_best.png``           – Baseline (config 1) vs las 3 mejores.
3. ``entropy_evolution.png``          – Tendencias de entropia para las 3 configuraciones.
4. ``hyperparam_comparison.png``      – Reward final agregado vs valores de hiperparametros.

Analisis de Evaluacion (si --evaluation-csv proporcionado):
5. ``evaluation_scores_ranking.png``  – Grafico de barras de scores promedio (pipes) de evaluacion.
6. ``evaluation_comparison.png``      – Reward de entrenamiento vs score de evaluacion.
7. ``training_vs_eval_top5.png``      – Curvas de entrenamiento con scores de evaluacion final.

ejecucion:
    python plot_learning_curves.py \
        --saved-models-dir savedModels \
        --output-dir plots/flappy_analysis \
        --max-config-id 18 \
        --evaluation-csv evaluation_top_models.csv
"""
from __future__ import annotations

import sys
import argparse
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

project_root = Path(__file__).parent.parent.parent
analysis_path = project_root / "scripts" / "analysis"
if str(analysis_path) not in sys.path:
    sys.path.insert(0, str(analysis_path))

import analyze_flappy_results as analyzer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SAVED_MODELS_DIR = BASE_DIR / "savedModels"
DEFAULT_OUTPUT_DIR = BASE_DIR / "plots" / "flappy_analysis"
ROLLING_WINDOW = 100
BASELINE_CONFIG_ID = 1


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PPO Flappy Bird learning curves and comparisons."
    )
    parser.add_argument(
        "--saved-models-dir",
        type=Path,
        default=DEFAULT_SAVED_MODELS_DIR,
        help="Directory with config_* folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store generated figures (default: %(default)s)",
    )
    parser.add_argument(
        "--max-config-id",
        type=int,
        default=18,
        help="Highest configuration id to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=ROLLING_WINDOW,
        help="Episodes used for smoothing/recent stats (default: %(default)s)",
    )
    parser.add_argument(
        "--entropy-window",
        type=int,
        default=200,
        help="Updates used for smoothing entropy curves (default: %(default)s)",
    )
    parser.add_argument(
        "--evaluation-csv",
        type=Path,
        default=None,
        help="Optional CSV with evaluation results (from evaluate_top_models.py)",
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
    return f"{int(row['config_id'])} | {row['config_name']}"


def prepare_series(df: pd.DataFrame, window: int) -> pd.Series:
    return df["reward"].rolling(window=window, min_periods=1).mean()


def collect_config_data(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    config_ids: Iterable[int],
    window: int,
) -> List[Tuple[pd.Series, pd.Series, str]]:
    """Return (episodes, rewards, label) tuples for the given config ids."""
    data: List[Tuple[pd.Series, pd.Series, str]] = []
    for config_id in config_ids:
        match = ranking_df.loc[ranking_df["config_id"] == config_id]
        if match.empty:
            print(f"[WARN] Config id {config_id} not found in ranking.")
            continue
        row = match.iloc[0]
        config_dir = analyzer.discover_config_dir(saved_models_dir, config_id)
        if config_dir is None:
            print(f"[WARN] Missing directory for config {config_id}.")
            continue
        try:
            rewards_df = load_rewards(config_dir)
        except FileNotFoundError:
            print(f"[WARN] Missing rewards.csv for {config_dir.name}.")
            continue
        if rewards_df.empty:
            print(f"[WARN] Rewards file is empty for {config_dir.name}.")
            continue
        smoothed = prepare_series(rewards_df, window=window)
        label = get_label(row)
        data.append((rewards_df["episode"], smoothed, label))
    return data


def plot_top5_learning_curves(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    output_dir: Path,
    window: int,
) -> None:
    top_configs = ranking_df.head(5)["config_id"].tolist()
    series = collect_config_data(
        ranking_df, saved_models_dir, top_configs, window
    )
    if not series:
        print("[WARN] Unable to plot top 5 learning curves.")
        return

    plt.figure(figsize=(12, 6))
    for episodes, rewards, label in series:
        plt.plot(episodes, rewards, label=label, linewidth=2)
    plt.xlabel("Episodio")
    plt.ylabel(f"Reward (media móvil {window})")
    plt.title("Curvas de aprendizaje – Mejores 5 configuraciones")
    plt.legend(loc="lower right")
    path = output_dir / "top5_learning_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado {path}")


def plot_baseline_vs_best(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    output_dir: Path,
    window: int,
) -> None:
    baseline_ids: List[int] = [BASELINE_CONFIG_ID]
    top3_ids = ranking_df.head(3)["config_id"].tolist()
    combined_ids = baseline_ids + [cid for cid in top3_ids if cid not in baseline_ids]

    series = collect_config_data(
        ranking_df, saved_models_dir, combined_ids, window
    )
    if not series:
        print("[WARN] Unable to plot baseline vs best.")
        return

    plt.figure(figsize=(12, 6))
    for episodes, rewards, label in series:
        plt.plot(episodes, rewards, label=label, linewidth=2)
    plt.xlabel("Episodio")
    plt.ylabel(f"Reward (media móvil {window})")
    plt.title("Baseline vs Mejores configuraciones")
    plt.legend(loc="lower right")
    path = output_dir / "baseline_vs_best.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado {path}")


def plot_entropy_evolution(
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    output_dir: Path,
    top_k: int,
    window: int,
) -> None:
    config_ids = ranking_df.head(top_k)["config_id"].tolist()
    plt.figure(figsize=(12, 6))
    plotted = False

    for config_id in config_ids:
        config_dir = analyzer.discover_config_dir(saved_models_dir, config_id)
        if config_dir is None:
            continue
        try:
            losses_df = load_losses(config_dir)
        except FileNotFoundError:
            print(f"[WARN] Missing losses.csv for {config_dir.name}.")
            continue
        if losses_df.empty:
            continue
        smooth_entropy = (
            (-losses_df["entropy_loss"])
            .rolling(window=window, min_periods=1)
            .mean()
        )
        label = get_label(
            ranking_df.loc[ranking_df["config_id"] == config_id].iloc[0]
        )
        plt.plot(
            losses_df["timestep"],
            smooth_entropy,
            label=label,
            linewidth=2,
        )
        plotted = True

    if not plotted:
        print("[WARN] Unable to plot entropy evolution.")
        return

    plt.xlabel("Timestep")
    plt.ylabel(f"Entropía (media móvil {window})")
    plt.title("Evolución de la entropía – Top 3 configuraciones")
    plt.legend(loc="upper right")
    path = output_dir / "entropy_evolution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado {path}")


def aggregate_by_hyperparam(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return mean/std of mean_reward_last grouped by ``column``."""
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


def add_value_labels(ax: plt.Axes) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3)


def plot_hyperparam_comparison(ranking_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    plots_info = [
        ("learning_rate", "Learning Rate"),
        ("hidden_size", "Hidden Size"),
        ("n_epochs", "Nº Epochs"),
        ("batch_size", "Batch Size"),
    ]

    for ax, (column, title) in zip(axes, plots_info):
        grouped = aggregate_by_hyperparam(ranking_df, column)
        if grouped.empty:
            ax.set_visible(False)
            continue
        ax.bar(grouped[column].astype(str), grouped["mean_reward"], color="#3b82f6")
        ax.set_title(title)
        ax.set_ylabel("Mean Reward (últimos 100)")
        add_value_labels(ax)

    fig.suptitle("Comparativa por hiperparámetros", fontsize=16)
    path = output_dir / "hyperparam_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado {path}")


def plot_evaluation_scores_ranking(eval_df: pd.DataFrame, output_dir: Path, top_k: int = 10) -> None:
    """Bar chart showing mean scores (pipes) from evaluation."""
    if eval_df.empty:
        print("[WARN] Evaluation dataframe is empty.")
        return
    
    df_top = eval_df.head(top_k).copy()
    df_top['short_name'] = df_top['config_name'].str.replace('_', ' ')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(range(len(df_top)))
    
    bars = ax.barh(df_top['short_name'], df_top['mean_score'], color=colors, alpha=0.8)
    
    # Añadir valores en las barras
    for i, (score, config_id) in enumerate(zip(df_top['mean_score'], df_top['config_id'])):
        ax.text(score + 2, i, f'{score:.1f} (#{int(config_id)})', 
                va='center', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Mean Score (Pipes Atravesados)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuración', fontsize=12, fontweight='bold')
    ax.set_title('Ranking de Evaluación - Scores Promedio por Configuración', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    path = output_dir / "evaluation_scores_ranking.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado {path}")


def plot_evaluation_comparison(
    eval_df: pd.DataFrame, 
    ranking_df: pd.DataFrame, 
    output_dir: Path,
) -> None:
    """Compare training final reward vs evaluation scores."""
    if eval_df.empty:
        print("[WARN] Evaluation dataframe is empty.")
        return
    
    merged = eval_df.merge(
        ranking_df[['config_id', 'mean_reward_last']], 
        on='config_id', 
        how='left'
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    merged_top = merged.head(10).copy()
    merged_top['short_name'] = merged_top['config_name'].str.replace('_', ' ')
    
    ax1.barh(merged_top['short_name'], merged_top['mean_score'], 
             color='skyblue', alpha=0.8, label='Evaluation Score (pipes)')
    ax1.set_xlabel('Mean Score (Pipes)', fontweight='bold')
    ax1.set_ylabel('Configuración', fontweight='bold')
    ax1.set_title('Score de Evaluación', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    ax2.barh(merged_top['short_name'], merged_top['mean_reward_last'], 
             color='coral', alpha=0.8, label='Training Reward')
    ax2.set_xlabel('Mean Reward (últimos 100 eps)', fontweight='bold')
    ax2.set_title('Reward de Entrenamiento', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    fig.suptitle('Comparación: Entrenamiento vs Evaluación', fontsize=16, fontweight='bold')
    
    path = output_dir / "evaluation_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado {path}")


def plot_training_with_eval_scores(
    eval_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    saved_models_dir: Path,
    output_dir: Path,
    window: int,
) -> None:
    """Show training curves for top 5 with their final evaluation scores annotated."""
    if eval_df.empty:
        print("[WARN] Evaluation dataframe is empty.")
        return
    
    top_configs = eval_df.head(5)['config_id'].tolist()
    series = collect_config_data(ranking_df, saved_models_dir, top_configs, window)
    
    if not series:
        print("[WARN] Unable to plot training vs eval.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for episodes, rewards, label in series:
        config_id = int(label.split(' | ')[0])
        eval_row = eval_df[eval_df['config_id'] == config_id]
        
        if not eval_row.empty:
            mean_score = eval_row.iloc[0]['mean_score']
            line, = ax.plot(episodes, rewards, label=label, linewidth=2)
            
            # Anotar el score final de evaluación
            last_episode = episodes.iloc[-1]
            last_reward = rewards.iloc[-1]
            ax.annotate(
                f'{mean_score:.0f} pipes',
                xy=(last_episode, last_reward),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color=line.get_color(),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=line.get_color(), alpha=0.8)
            )
    
    ax.set_xlabel('Episodio', fontweight='bold')
    ax.set_ylabel(f'Reward (media móvil {window})', fontweight='bold')
    ax.set_title('Curvas de Entrenamiento con Scores de Evaluación Final', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    path = output_dir / "training_vs_eval_top5.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado {path}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    saved_models_dir = args.saved_models_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ensure_output_dir(output_dir)

    sns.set_theme(style="whitegrid")

    ranking_df, missing = analyzer.analyze_configs(
        saved_models_dir=saved_models_dir,
        max_config_id=args.max_config_id,
        rolling_window=args.rolling_window,
    )

    if ranking_df.empty:
        print("[ERROR] No se pudieron generar estadísticas para las configuraciones.")
        if missing:
            print("Problemas detectados:")
            for issue in missing:
                print(f"  - {issue}")
        return 1

    if missing:
        print("Advertencias durante el análisis:")
        for issue in missing:
            print(f"  - {issue}")

    print("\n[INFO] Generando gráficas de análisis de entrenamiento...")
    plot_top5_learning_curves(
        ranking_df,
        saved_models_dir,
        output_dir,
        window=args.rolling_window,
    )
    plot_baseline_vs_best(
        ranking_df,
        saved_models_dir,
        output_dir,
        window=args.rolling_window,
    )
    plot_entropy_evolution(
        ranking_df,
        saved_models_dir,
        output_dir,
        top_k=3,
        window=args.entropy_window,
    )
    plot_hyperparam_comparison(ranking_df, output_dir)

    generated_files = [
        "top5_learning_curves.png",
        "baseline_vs_best.png",
        "entropy_evolution.png",
        "hyperparam_comparison.png",
    ]

    if args.evaluation_csv is not None:
        eval_csv_path = args.evaluation_csv.expanduser().resolve()
        if eval_csv_path.exists():
            print(f"\n[INFO] Cargando resultados de evaluación desde {eval_csv_path}")
            try:
                eval_df = pd.read_csv(eval_csv_path)
                print(f"[INFO] Generando gráficas de evaluación...")
                
                plot_evaluation_scores_ranking(eval_df, output_dir, top_k=10)
                plot_evaluation_comparison(eval_df, ranking_df, output_dir)
                plot_training_with_eval_scores(
                    eval_df, ranking_df, saved_models_dir, output_dir, 
                    window=args.rolling_window
                )
                
                generated_files.extend([
                    "evaluation_scores_ranking.png",
                    "evaluation_comparison.png",
                    "training_vs_eval_top5.png",
                ])
            except Exception as e:
                print(f"[WARN] Error al procesar evaluación: {e}")
        else:
            print(f"[WARN] Archivo de evaluación no encontrado: {eval_csv_path}")

    print("\n" + "="*80)
    print(f"Figuras generadas en: {output_dir}")
    for filename in generated_files:
        print(f"  - {filename}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

