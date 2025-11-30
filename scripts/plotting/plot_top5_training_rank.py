#!/usr/bin/env python3
"""
graficar las 5 mejores configuraciones segun el ranking de entrenamiento
(basado en mean reward de los ultimos 100 episodios).

Top 5 segun entrenamiento:
1. Config 14 - Red_Grande_Menos_Epocas_LR_Suave (181.62)
2. Config 9  - Red_Grande_Batch_Grande (167.06)
3. Config 13 - Red_Grande_Una_Epoca_Batch_Grande (156.76)
4. Config 11 - Red_Grande_Una_Epoca (148.05)
5. Config 12 - Red_Grande_Menos_Epocas_Batch_Grande (144.58)

ejecucion:
    python plot_top5_training_rank.py --saved-models-dir savedModels --output-dir plots
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def discover_config_dir(saved_models_dir: Path, config_id: int) -> Optional[Path]:
    explicit = saved_models_dir / f"config_{config_id}"
    if explicit.exists():
        return explicit
    matches = sorted(saved_models_dir.glob(f"config_{config_id}_*"))
    return matches[0] if matches else None


def load_rewards(config_dir: Path) -> pd.DataFrame:
    rewards_path = config_dir / "logs" / "rewards.csv"
    df = pd.read_csv(rewards_path)
    return df.sort_values("timestep").reset_index(drop=True)


def plot_top5_training(
    saved_models_dir: Path,
    output_path: Path,
    rolling_window: int = 100
) -> None:
    
    # Top 5 configuraciones segun el ranking de entrenamiento
    top5_configs = [
        (14, "Red_Grande_Menos_Epocas_LR_Suave", 181.62),
        (9, "Red_Grande_Batch_Grande", 167.06),
        (13, "Red_Grande_Una_Epoca_Batch_Grande", 156.76),
        (11, "Red_Grande_Una_Epoca", 148.05),
        (12, "Red_Grande_Menos_Epocas_Batch_Grande", 144.58),
    ]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
    
    for idx, (config_id, config_name, mean_reward) in enumerate(top5_configs):
        config_dir = discover_config_dir(saved_models_dir, config_id)
        
        if config_dir is None:
            print(f"[WARN] No se encontró config {config_id}")
            continue
        
        try:
            df = load_rewards(config_dir)
        except FileNotFoundError:
            print(f"[WARN] No se encontró rewards.csv para config {config_id}")
            continue
        
        df['reward_smooth'] = df['reward'].rolling(
            window=rolling_window, 
            min_periods=1
        ).mean()
        
        label = f"#{idx+1} | ID {config_id} | {config_name.replace('_', ' ')}\n(Mean: {mean_reward:.1f})"
        line = ax.plot(
            df['timestep'],
            df['reward_smooth'],
            label=label,
            linewidth=2.5,
            color=colors[idx],
            alpha=0.9
        )[0]
        
        final_episode = df['timestep'].iloc[-1]
        final_reward = df['reward_smooth'].iloc[-1]
        
        ax.annotate(
            f'{final_reward:.0f}',
            xy=(final_episode, final_reward),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color=colors[idx],
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor=colors[idx],
                alpha=0.8
            )
        )
    
    ax.set_xlabel('Total Timesteps', fontsize=13, fontweight='bold')
    ax.set_ylabel(
        f'Reward (media móvil {rolling_window} episodios)',
        fontsize=13,
        fontweight='bold'
    )
    ax.set_title(
        'Top 5 Configuraciones según Ranking de Entrenamiento\n' +
        'Ordenadas por Mean Reward (últimos 100 episodios)',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(
        loc='lower right',
        fontsize=9,
        framealpha=0.95,
        ncol=1
    )
    
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    
    ax.text(
        0.02, 0.98,
        'Ranking basado en reward promedio\nde los últimos 100 episodios',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='lightblue',
            alpha=0.2,
            edgecolor='black'
        )
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Gráfico guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Graficar top 5 configuraciones según ranking de entrenamiento'
    )
    parser.add_argument(
        '--saved-models-dir',
        type=Path,
        default=Path('savedModels'),
        help='Directorio con carpetas config_*'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('plots/flappy_analysis'),
        help='Directorio de salida para la figura'
    )
    parser.add_argument(
        '--rolling-window',
        type=int,
        default=100,
        help='Ventana para media móvil'
    )
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = args.output_dir / 'top5_training_ranking.png'
    
    sns.set_theme(style="whitegrid")
    
    plot_top5_training(
        saved_models_dir=args.saved_models_dir,
        output_path=output_path,
        rolling_window=args.rolling_window
    )
    
    print("\n" + "="*80)
    print(f"Figura generada: {output_path}")
    print("\nTop 5 por Reward de Entrenamiento:")
    print("  1. Config 14 - Red_Grande_Menos_Epocas_LR_Suave (181.62)")
    print("  2. Config 9  - Red_Grande_Batch_Grande (167.06)")
    print("  3. Config 13 - Red_Grande_Una_Epoca_Batch_Grande (156.76)")
    print("  4. Config 11 - Red_Grande_Una_Epoca (148.05)")
    print("  5. Config 12 - Red_Grande_Menos_Epocas_Batch_Grande (144.58)")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

