#!/usr/bin/env python3
"""
generar gráfico comparativo crítico entre Config 4 (1M timesteps) y Config 20 (8M timesteps)
para mostrar visualmente la degradación de performance con entrenamiento extendido.

ejecucion:
    python plot_config4_vs_config20.py --saved-models-dir savedModels --output-dir plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def discover_config_dir(saved_models_dir: Path, config_id: int) -> Path:
    """
    encontrar directorio de configuracion por ID.
    
    """
    explicit = saved_models_dir / f"config_{config_id}"
    if explicit.exists():
        return explicit
    
    matches = sorted(saved_models_dir.glob(f"config_{config_id}_*"))
    if matches:
        return matches[0]
    
    raise FileNotFoundError(f"No se encontró configuración {config_id} en {saved_models_dir}")


def load_rewards(config_dir: Path) -> pd.DataFrame:
    """cargar rewards.csv de un directorio de configuracion."""
    rewards_path = config_dir / "logs" / "rewards.csv"
    df = pd.read_csv(rewards_path)
    return df.sort_values("episode").reset_index(drop=True)

def plot_config_comparison(
    config4_dir: Path,
    config20_dir: Path,
    output_path: Path,
    rolling_window: int = 100
) -> None:
    """graficar Config 4 vs Config 20 con anotaciones de degradacion."""
    
    # cargar datos
    df4 = load_rewards(config4_dir)
    df20 = load_rewards(config20_dir)
    
    # calcular media movil
    df4['reward_smooth'] = df4['reward'].rolling(window=rolling_window, min_periods=1).mean()
    df20['reward_smooth'] = df20['reward'].rolling(window=rolling_window, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    line4 = ax.plot(
        df4['episode'], 
        df4['reward_smooth'], 
        label='Config 4 (1M timesteps) - batch=64, epochs=4',
        linewidth=2.5,
        color='#2E86AB',  
        alpha=0.9
    )[0]
    
    line20 = ax.plot(
        df20['episode'], 
        df20['reward_smooth'], 
        label='Config 20 (8M timesteps) - extensión Config 4',
        linewidth=2.5,
        color='#A23B72',  
        alpha=0.9
    )[0]
    
    final_reward_4 = df4['reward_smooth'].iloc[-1]
    final_episode_4 = df4['episode'].iloc[-1]
    
    ax.annotate(
        f'Convergencia temprana\nReward: {final_reward_4:.1f}\nEval: 219.4 pipes',
        xy=(final_episode_4, final_reward_4),
        xytext=(final_episode_4 + 50, final_reward_4 + 50),
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#2E86AB', edgecolor='black', 
                  alpha=0.2, linewidth=1.5),
        arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'),
        fontsize=11,
        fontweight='bold',
        color='#2E86AB'
    )
    
    final_reward_20 = df20['reward_smooth'].iloc[-1]
    final_episode_20 = df20['episode'].iloc[-1]
    
    ax.annotate(
        f'Tras 8M timesteps\nReward: {final_reward_20:.1f}\nEval: 177.1 pipes\n⚠️ Degradación -19.3%',
        xy=(final_episode_20, final_reward_20),
        xytext=(final_episode_20 - 300, final_reward_20 - 100),
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#A23B72', edgecolor='black', 
                  alpha=0.2, linewidth=1.5),
        arrowprops=dict(arrowstyle='->', lw=2, color='#A23B72'),
        fontsize=11,
        fontweight='bold',
        color='#A23B72'
    )
    
    ax.axvline(
        x=final_episode_4, 
        color='gray', 
        linestyle='--', 
        linewidth=1.5, 
        alpha=0.7,
        label=f'Fin Config 4 (~{final_episode_4:.0f} episodios)'
    )
    
    ax.set_xlabel('Episodio', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Reward (media móvil {rolling_window} episodios)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Hallazgo Crítico: Config 4 vs Config 20\n' + 
        'Degradación de Performance con Entrenamiento Extendido',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    
    ax.text(
        0.02, 0.98,
        'Más entrenamiento ≠ Mejor performance\n' +
        'Config 4 convergió óptimamente en 1M timesteps',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.15, edgecolor='black')
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Gráfico guardado en: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Generar gráfico comparativo Config 4 vs Config 20'
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
        default=Path('plots'),
        help='Directorio de salida para la figura'
    )
    parser.add_argument(
        '--rolling-window',
        type=int,
        default=100,
        help='Ventana para media móvil'
    )
    
    args = parser.parse_args()
    
    try:
        config4_dir = discover_config_dir(args.saved_models_dir, 4)
        print(f"[INFO] Encontrado Config 4: {config4_dir.name}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    
    try:
        config20_dir = discover_config_dir(args.saved_models_dir, 20)
        print(f"[INFO] Encontrado Config 20: {config20_dir.name}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = args.output_dir / 'config4_vs_config20_degradation.png'
    
    sns.set_theme(style="whitegrid")
    
    plot_config_comparison(
        config4_dir=config4_dir,
        config20_dir=config20_dir,
        output_path=output_path,
        rolling_window=args.rolling_window
    )
    
    return 0

if __name__ == '__main__':
    raise SystemExit(main())