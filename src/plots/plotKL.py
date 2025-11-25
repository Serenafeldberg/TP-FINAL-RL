"""
Script para graficar y comparar approx_kl de múltiples configuraciones.

Muestra cómo evoluciona la divergencia KL durante el entrenamiento
para diferentes hiperparámetros.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_kl_comparison(config_names=None, window_size=10, save_path=None):
    """
    Graficar approx_kl de múltiples configuraciones.
    
    Args:
        config_names: Lista de nombres de configuraciones (ej: ["config_8_Red_Grande_Menos_Epocas", ...])
                     Si None, intenta cargar todas las disponibles.
        window_size: Tamaño de ventana para suavizado (media móvil)
        save_path: Ruta para guardar la figura (opcional)
    """
    
    if config_names is None:
        # Configuraciones por defecto
        config_names = [
            "config_12_Red_Grande_Menos_Epocas_Batch_Grande",
            "config_13_Red_Grande_Una_Epoca_Batch_Grande",
            "config_15_Red_Gigante_Una_Epoca",
            "config_9_Red_Grande_Batch_Grande",
        ]
    
    colors = ['blue', 'red', 'green', "orange"]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    config_stats = []
    
    for config_name, color in zip(config_names, colors):
        # Ruta al CSV de losses
        losses_path = Path(__file__).parent.parent.parent / "savedModels" / config_name / "logs" / "losses.csv"
        
        if not losses_path.exists():
            print(f"⚠️  No se encontró: {losses_path}")
            continue
        
        try:
            losses_df = pd.read_csv(losses_path)
            timesteps = losses_df['timestep'].values
            approx_kl = losses_df['approx_kl'].values
            
            # Aplicar suavizado (media móvil)
            if len(approx_kl) >= window_size:
                smoothed_kl = pd.Series(approx_kl).rolling(window=window_size, min_periods=1).mean().values
            else:
                smoothed_kl = approx_kl
            
            # Graficar
            ax.plot(timesteps, approx_kl, alpha=0.2, color=color, linewidth=0.5)  # Raw (semi-transparente)
            ax.plot(timesteps, smoothed_kl, color=color, linewidth=2.5, 
                   label=f"{config_name} (MA={window_size})", alpha=0.8)
            
            # Estadísticas
            mean_kl = np.mean(approx_kl)
            max_kl = np.max(approx_kl)
            final_kl = approx_kl[-1] if len(approx_kl) > 0 else 0
            
            config_stats.append({
                'config': config_name,
                'mean_kl': mean_kl,
                'max_kl': max_kl,
                'final_kl': final_kl,
                'n_updates': len(approx_kl)
            })
            
            print(f"✓ {config_name}:")
            print(f"    Updates: {len(approx_kl)}")
            print(f"    Mean KL: {mean_kl:.6f}")
            print(f"    Max KL: {max_kl:.6f}")
            print(f"    Final KL: {final_kl:.6f}")
            
        except Exception as e:
            print(f"❌ Error al procesar {config_name}: {e}")
    
    # Configurar el gráfico
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Approximate KL Divergence', fontsize=12)
    ax.set_title('Comparación de Divergencia KL entre Configuraciones', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Línea de referencia (threshold típico de PPO)
    ax.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Umbral típico (0.01)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfico guardado en: {save_path}")
    
    # Mostrar tabla comparativa
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO DE KL DIVERGENCE")
    print("="*80)
    if config_stats:
        stats_df = pd.DataFrame(config_stats)
        print(stats_df.to_string(index=False))
        print("="*80)
    
    return fig


def plot_kl_single(config_name, save_path=None):
    """
    Graficar approx_kl de una sola configuración con más detalle.
    
    Args:
        config_name: Nombre de la configuración
        save_path: Ruta para guardar la figura (opcional)
    """
    losses_path = Path(__file__).parent.parent.parent / "savedModels" / config_name / "logs" / "losses.csv"
    
    if not losses_path.exists():
        print(f"❌ Error: No se encontró {losses_path}")
        return
    
    losses_df = pd.read_csv(losses_path)
    timesteps = losses_df['timestep'].values
    approx_kl = losses_df['approx_kl'].values
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: KL raw + suavizado
    ax1 = axes[0]
    ax1.plot(timesteps, approx_kl, alpha=0.3, color='blue', label='Raw approx_kl')
    
    # Suavizado con diferentes ventanas
    for window in [5, 10, 20]:
        if len(approx_kl) >= window:
            smoothed = pd.Series(approx_kl).rolling(window=window, min_periods=1).mean().values
            ax1.plot(timesteps, smoothed, linewidth=2, label=f'MA({window})', alpha=0.7)
    
    ax1.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Umbral (0.01)')
    ax1.set_xlabel('Timestep', fontsize=11)
    ax1.set_ylabel('Approx KL', fontsize=11)
    ax1.set_title(f'{config_name} - Evolución de KL Divergence', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Histograma de distribución
    ax2 = axes[1]
    ax2.hist(approx_kl, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(approx_kl), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(approx_kl):.6f}')
    ax2.axvline(x=0.01, color='orange', linestyle='--', linewidth=2, label='Umbral: 0.01')
    ax2.set_xlabel('Approx KL value', fontsize=11)
    ax2.set_ylabel('Frecuencia', fontsize=11)
    ax2.set_title('Distribución de valores KL', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    # Estadísticas
    print("\n" + "="*60)
    print(f"ESTADÍSTICAS DE KL - {config_name}")
    print("="*60)
    print(f"Total de updates: {len(approx_kl)}")
    print(f"Mean KL: {np.mean(approx_kl):.8f}")
    print(f"Std KL: {np.std(approx_kl):.8f}")
    print(f"Min KL: {np.min(approx_kl):.8f}")
    print(f"Max KL: {np.max(approx_kl):.8f}")
    print(f"Median KL: {np.median(approx_kl):.8f}")
    print(f"% updates con KL < 0.01: {(approx_kl < 0.01).sum() / len(approx_kl) * 100:.1f}%")
    print(f"% updates con KL > 0.1: {(approx_kl > 0.1).sum() / len(approx_kl) * 100:.1f}%")
    print("="*60 + "\n")
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Graficar Approx KL de configuraciones")
    parser.add_argument("--config", type=str, default=None,
                       help="Nombre de configuración específica (si no, compara varias)")
    parser.add_argument("--compare", action="store_true",
                       help="Modo comparación (default si no se especifica --config)")
    parser.add_argument("--window", type=int, default=10,
                       help="Tamaño de ventana para media móvil (default: 10)")
    
    args = parser.parse_args()
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("PLOTEO DE APPROX KL DIVERGENCE")
    print("=" * 80 + "\n")
    
    if args.config:
        # Modo single: analizar una configuración en detalle
        print(f"Analizando configuración: {args.config}\n")
        fig = plot_kl_single(
            config_name=args.config,
            save_path=output_dir / f"kl_single_{args.config}.png"
        )
    else:
        # Modo comparación: varias configuraciones
        print("Comparando múltiples configuraciones\n")
        fig = plot_kl_comparison(
            window_size=args.window,
            save_path=output_dir / "kl_comparison.png"
        )
    
    print("\n✓ Listo. Mostrando gráficos...")
    plt.show()
