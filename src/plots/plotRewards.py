"""
Script para visualizar las métricas de entrenamiento de PPO.

Grafica:
- Recompensas por episodio (con promedio móvil)
- Longitud de episodios
- Pérdidas (policy, value, entropy)
- Métricas adicionales (clip fraction, KL divergence, learning rate)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuración de rutas
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
REWARDS_CSV = PROJECT_ROOT / "logs" / "rewards.csv"
LOSSES_CSV = PROJECT_ROOT / "logs" / "losses.csv"


def moving_average(data, window_size=100):
    """Calcular promedio móvil."""
    if len(data) < window_size:
        return data
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values


def plot_rewards(rewards_df, window_size=100, save_path=None):
    """
    Graficar recompensas por episodio.
    
    Args:
        rewards_df: DataFrame con columnas timestep, episode, reward, length
        window_size: tamaño de ventana para promedio móvil
        save_path: ruta para guardar la figura (opcional)
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gráfico 1: Recompensas
    ax1 = axes[0]
    episodes = rewards_df['episode'].values
    rewards = rewards_df['reward'].values
    
    # Recompensas individuales (transparentes)
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Recompensa por episodio')
    
    # Promedio móvil
    if len(rewards) >= window_size:
        ma_rewards = moving_average(rewards, window_size)
        ax1.plot(episodes, ma_rewards, color='red', linewidth=2, 
                label=f'Promedio móvil ({window_size} episodios)')
    
    ax1.set_xlabel('Episodio', fontsize=12)
    ax1.set_ylabel('Recompensa', fontsize=12)
    ax1.set_title('Recompensas por Episodio', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Estadísticas en el gráfico
    if len(rewards) > 0:
        final_avg = np.mean(rewards[-window_size:]) if len(rewards) >= window_size else np.mean(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        ax1.text(0.02, 0.98, 
                f'Promedio final ({window_size} últimos): {final_avg:.2f}\n'
                f'Máxima: {max_reward:.2f}\n'
                f'Mínima: {min_reward:.2f}',
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Gráfico 2: Longitud de episodios
    ax2 = axes[1]
    lengths = rewards_df['length'].values
    
    ax2.plot(episodes, lengths, alpha=0.3, color='green', label='Longitud por episodio')
    
    if len(lengths) >= window_size:
        ma_lengths = moving_average(lengths, window_size)
        ax2.plot(episodes, ma_lengths, color='orange', linewidth=2,
                label=f'Promedio móvil ({window_size} episodios)')
    
    ax2.set_xlabel('Episodio', fontsize=12)
    ax2.set_ylabel('Longitud del Episodio', fontsize=12)
    ax2.set_title('Longitud de Episodios', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def plot_losses(losses_df, save_path=None):
    """
    Graficar pérdidas y métricas de entrenamiento.
    
    Args:
        losses_df: DataFrame con columnas timestep, policy_loss, value_loss, etc.
        save_path: ruta para guardar la figura (opcional)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    timesteps = losses_df['timestep'].values
    
    # 1. Policy Loss
    ax = axes[0, 0]
    ax.plot(timesteps, losses_df['policy_loss'], color='blue', linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Policy Loss', fontsize=10)
    ax.set_title('Policy Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Value Loss
    ax = axes[0, 1]
    ax.plot(timesteps, losses_df['value_loss'], color='red', linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Value Loss', fontsize=10)
    ax.set_title('Value Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Escala logarítmica si los valores son muy grandes
    if losses_df['value_loss'].max() > 100:
        ax.set_yscale('log')
    
    # 3. Entropy Loss (negativo de la entropía)
    ax = axes[1, 0]
    entropy = -losses_df['entropy_loss']  # Convertir a entropía positiva
    ax.plot(timesteps, entropy, color='green', linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Entropy', fontsize=10)
    ax.set_title('Entropy (Exploración)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Clip Fraction
    ax = axes[1, 1]
    ax.plot(timesteps, losses_df['clip_fraction'], color='orange', linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Clip Fraction', fontsize=10)
    ax.set_title('Clip Fraction (Ratio de clipping)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 5. Approximate KL Divergence
    ax = axes[2, 0]
    ax.plot(timesteps, losses_df['approx_kl'], color='purple', linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Approx KL', fontsize=10)
    ax.set_title('Approximate KL Divergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Línea de referencia para KL alto (indica actualizaciones muy grandes)
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Umbral (0.01)')
    ax.legend(fontsize=8)
    
    # 6. Learning Rate
    ax = axes[2, 1]
    ax.plot(timesteps, losses_df['learning_rate'], color='brown', linewidth=1.5)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Learning Rate', fontsize=10)
    ax.set_title('Learning Rate (con decay)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    return fig


def plot_combined_summary(rewards_df, losses_df, window_size=100, save_path=None):
    """
    Crear un resumen combinado con las métricas más importantes.
    
    Args:
        rewards_df: DataFrame de recompensas
        losses_df: DataFrame de pérdidas
        window_size: tamaño de ventana para promedio móvil
        save_path: ruta para guardar la figura (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Recompensas (arriba izquierda)
    ax = axes[0, 0]
    episodes = rewards_df['episode'].values
    rewards = rewards_df['reward'].values
    ax.plot(episodes, rewards, alpha=0.3, color='blue')
    if len(rewards) >= window_size:
        ma_rewards = moving_average(rewards, window_size)
        ax.plot(episodes, ma_rewards, color='red', linewidth=2,
                label=f'Promedio móvil ({window_size})')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa')
    ax.set_title('Recompensas', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Policy Loss (arriba derecha)
    ax = axes[0, 1]
    timesteps = losses_df['timestep'].values
    ax.plot(timesteps, losses_df['policy_loss'], color='blue', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Value Loss (abajo izquierda)
    ax = axes[1, 0]
    ax.plot(timesteps, losses_df['value_loss'], color='red', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value Loss')
    ax.set_title('Value Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)
    if losses_df['value_loss'].max() > 100:
        ax.set_yscale('log')
    
    # 4. Entropy (abajo derecha)
    ax = axes[1, 1]
    entropy = -losses_df['entropy_loss']
    ax.plot(timesteps, entropy, color='green', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy (Exploración)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Resumen guardado en: {save_path}")
    
    return fig


def main():
    """Función principal."""
    print("=" * 60)
    print("VISUALIZACIÓN DE MÉTRICAS DE ENTRENAMIENTO PPO")
    print("=" * 60)
    
    # Verificar que los archivos existan
    if not REWARDS_CSV.exists():
        print(f"❌ Error: No se encontró {REWARDS_CSV}")
        return
    
    if not LOSSES_CSV.exists():
        print(f"❌ Error: No se encontró {LOSSES_CSV}")
        return
    
    # Leer datos
    print(f"\n📊 Leyendo datos...")
    rewards_df = pd.read_csv(REWARDS_CSV)
    losses_df = pd.read_csv(LOSSES_CSV)
    
    print(f"  ✓ Recompensas: {len(rewards_df)} episodios")
    print(f"  ✓ Pérdidas: {len(losses_df)} updates")
    
    # Crear directorio de salida si no existe
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Graficar recompensas
    print(f"\n📈 Generando gráfico de recompensas...")
    fig_rewards = plot_rewards(rewards_df, window_size=100, 
                              save_path=output_dir / "rewards.png")
    
    # Graficar pérdidas
    print(f"\n📉 Generando gráfico de pérdidas...")
    fig_losses = plot_losses(losses_df, 
                            save_path=output_dir / "losses.png")
    
    # Resumen combinado
    print(f"\n📊 Generando resumen combinado...")
    fig_summary = plot_combined_summary(rewards_df, losses_df, window_size=100,
                                        save_path=output_dir / "summary.png")
    
    print(f"\n✅ Visualizaciones completadas!")
    print(f"   Archivos guardados en: {output_dir}")
    print(f"\n   - rewards.png: Recompensas y longitudes")
    print(f"   - losses.png: Todas las pérdidas y métricas")
    print(f"   - summary.png: Resumen combinado")
    
    # Mostrar estadísticas
    print(f"\n📊 Estadísticas:")
    print(f"   Recompensas:")
    print(f"     - Promedio: {rewards_df['reward'].mean():.2f}")
    print(f"     - Máxima: {rewards_df['reward'].max():.2f}")
    print(f"     - Mínima: {rewards_df['reward'].min():.2f}")
    print(f"     - Últimos 100 episodios: {rewards_df['reward'].tail(100).mean():.2f}")
    
    print(f"\n   Pérdidas:")
    print(f"     - Policy Loss (promedio): {losses_df['policy_loss'].mean():.4f}")
    print(f"     - Value Loss (promedio): {losses_df['value_loss'].mean():.4f}")
    print(f"     - Entropy (promedio): {-losses_df['entropy_loss'].mean():.4f}")
    print(f"     - Clip Fraction (promedio): {losses_df['clip_fraction'].mean():.4f}")
    
    # Mostrar gráficos
    plt.show()


if __name__ == "__main__":
    main()

