#!/usr/bin/env python3
"""
Script para graficar la entropía de cada configuración PPO desde los archivos losses.csv.
La entropía se calcula como -entropy_loss (ya que entropy_loss es el negativo de la entropía).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_losses_data(config_dirs, max_timesteps=1_000_000):
    """
    Carga los datos de losses.csv de todas las configuraciones.
    
    Args:
        config_dirs: Lista de directorios de configuraciones (Path objects)
        max_timesteps: Máximo de timesteps a incluir (para normalizar)
    
    Returns:
        Dict con DataFrames de cada configuración
    """
    losses_data = {}
    
    for config_dir in config_dirs:
        config_name = config_dir.name
        losses_file = config_dir / "logs" / "losses.csv"
        
        if losses_file.exists():
            try:
                df = pd.read_csv(losses_file)
                # Filtrar hasta max_timesteps
                df = df[df['timestep'] <= max_timesteps]
                
                if not df.empty:
                    # Calcular entropía como -entropy_loss
                    df['entropy'] = -df['entropy_loss']
                    losses_data[config_name] = df
                    print(f"✓ Cargado {config_name}: {len(df)} puntos de datos")
                else:
                    print(f"{config_name}: Sin datos dentro del rango de timesteps")
            except Exception as e:
                print(f"Error cargando {config_name}: {e}")
        else:
            print(f"No encontrado: {losses_file}")
    
    return losses_data

def plot_entropy_comparison(losses_data):
    """
    Crea gráfico de comparación de entropía entre configuraciones.
    """
    plt.figure(figsize=(15, 10))
    
    # Colores distintos para cada configuración
    colors = plt.cm.tab20(range(len(losses_data)))
    
    for i, (config_name, df) in enumerate(losses_data.items()):
        # Extraer número de configuración para el label
        config_id = config_name.split('_')[1] if 'config_' in config_name else config_name
        label = f"Config {config_id}"
        
        plt.plot(df['timestep'], df['entropy'], 
                label=label, 
                color=colors[i], 
                alpha=0.8,
                linewidth=1.5)
    
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.title('Entropy Evolution - PPO Flappy Bird Training\nComparison of All Configurations', 
              fontsize=14, fontweight='bold')
    
    # Mejorar la leyenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Mejorar el grid
    plt.grid(True, alpha=0.3)
    
    # Formato de ejes
    plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Ajustar layout para que la leyenda no se corte
    plt.tight_layout()
    
    return plt

def plot_entropy_subplots(losses_data):
    """
    Crea subplots individuales para mejor visualización de cada configuración.
    """
    n_configs = len(losses_data)
    cols = 3
    rows = (n_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    # Aplanar axes para fácil iteración
    flat_axes = [ax for row in axes for ax in row]
    
    for i, (config_name, df) in enumerate(losses_data.items()):
        ax = flat_axes[i]
        
        # Extraer número de configuración
        config_id = config_name.split('_')[1] if 'config_' in config_name else config_name
        title = f"Config {config_id}: {config_name.replace('config_' + config_id + '_', '')}"
        
        ax.plot(df['timestep'], df['entropy'], 
               color=plt.cm.tab20(i), 
               linewidth=2,
               alpha=0.8)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Timesteps', fontsize=10)
        ax.set_ylabel('Entropy', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Ocultar subplots vacíos
    for i in range(len(losses_data), len(flat_axes)):
        flat_axes[i].set_visible(False)
    
    plt.suptitle('Entropy Evolution - Individual Configurations', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def main():
    """Función principal"""
    print("Iniciando análisis de entropía...")
    
    # Directorio base de modelos guardados - usar ruta absoluta como en plotRewards.py
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # src/plots -> src -> project_root
    base_dir = project_root / "savedModels"
    
    print(f"Buscando en: {base_dir.absolute()}")
    
    # Configuraciones específicas que queremos plotear
    target_configs = [15, 9, 8, 11, 13]
    
    # Buscar solo las configuraciones especificadas
    config_dirs = []
    if base_dir.exists():
        for config_id in target_configs:
            # Buscar carpetas que empiecen con config_{id}_
            for item in base_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"config_{config_id}_"):
                    config_dirs.append(item)
                    break  # Solo tomar la primera que coincida
    
    if not config_dirs:
        print(f"No se encontraron las configuraciones {target_configs} en {base_dir}")
        print(f"Contenido del directorio:")
        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir() and item.name.startswith("config_"):
                    print(f"   • {item.name}")
        else:
            print(f"   El directorio {base_dir} no existe")
        return
    
    print(f"Encontradas {len(config_dirs)} de las configuraciones solicitadas {target_configs}:")
    for config_dir in sorted(config_dirs):
        print(f"   • {config_dir.name}")
    
    # Cargar datos
    print(f"\nCargando datos (máximo 1M timesteps)...")
    losses_data = load_losses_data(sorted(config_dirs), max_timesteps=1_000_000)
    
    if not losses_data:
        print("No se pudieron cargar datos de ninguna configuración")
        return
    
    print(f"\nDatos cargados para {len(losses_data)} configuraciones")
    
    # Crear directorio de plots si no existe
    plots_dir = Path(".")
    plots_dir.mkdir(exist_ok=True)
    
    # Gráfico de comparación general
    print("\nGenerando gráfico de comparación general...")
    plt1 = plot_entropy_comparison(losses_data)
    
    # Guardar gráfico
    output_file1 = plots_dir / "entropy_comparison_all_configs.png"
    plt1.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"Guardado: {output_file1}")
    
    # Gráfico de subplots individuales
    print("\nGenerando gráficos individuales...")
    fig2 = plot_entropy_subplots(losses_data)
    
    # Guardar gráfico
    output_file2 = plots_dir / "entropy_individual_configs.png"
    fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Guardado: {output_file2}")
    
    # Mostrar estadísticas básicas
    print(f"\nEstadísticas de entropía:")
    print("-" * 60)
    for config_name, df in losses_data.items():
        config_id = config_name.split('_')[1] if 'config_' in config_name else config_name
        entropy_mean = df['entropy'].mean()
        entropy_final = df['entropy'].iloc[-1] if len(df) > 0 else 0
        entropy_max = df['entropy'].max()
        entropy_min = df['entropy'].min()
        
        print(f"Config {config_id:2s}: Mean={entropy_mean:6.3f} | "
              f"Final={entropy_final:6.3f} | "
              f"Max={entropy_max:6.3f} | "
              f"Min={entropy_min:6.3f}")
    
    print(f"\nAnálisis completado. Archivos guardados en: {plots_dir.absolute()}")
    
    # Mostrar gráficos (opcional)
    try:
        plt.show()
    except:
        print("(No se pueden mostrar gráficos en este entorno)")

if __name__ == "__main__":
    main()
