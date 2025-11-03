import os
import random
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime


#REPRODUCIBILIDAD
def set_seed(seed: int) -> None:
    """
    Setea la seed para reproducibilidad en todos los frameworks.
    
    Args:
        seed: Seed para random, numpy, torch, etc.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Para determinismo completo (puede ser más lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed}")


# LOGGING Y MÉTRICAS
class Logger:
    """
    Logger simple para trackear métricas durante el entrenamiento.
    Guarda en CSV y permite imprimir estadísticas.
    """
    
    def __init__(self, log_dir: str, log_name: str = "training_log"):
        """
        Args:
            log_dir: Directorio donde guardar los logs
            log_name: Nombre base del archivo de log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivo de log con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{log_name}_{timestamp}.csv"
        
        # Buffer para acumular métricas
        self.metrics_buffer: Dict[str, List[float]] = {}
        
        print(f"✓ Logger initialized: {self.log_file}")
    
    def log(self, metrics: Dict[str, float], step: int) -> None:
        """
        Loggea métricas para un step específico.
        
        Args:
            metrics: Diccionario de métricas (e.g., {"reward": 10.5, "loss": 0.3})
            step: Paso de entrenamiento
        """
        # Agregar step a las métricas
        metrics_with_step = {"step": step, **metrics}
        
        # Acumular en buffer
        for key, value in metrics_with_step.items():
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []
            self.metrics_buffer[key].append(value)
        
        # Escribir en archivo
        self._write_to_file(metrics_with_step)
    
    def _write_to_file(self, metrics: Dict[str, float]) -> None:
        """Escribe métricas al archivo CSV."""
        # Si el archivo no existe, crear header
        file_exists = self.log_file.exists()
        
        with open(self.log_file, 'a') as f:
            if not file_exists:
                # Escribir header
                header = ",".join(metrics.keys())
                f.write(header + "\n")
            
            # Escribir valores
            values = ",".join(str(v) for v in metrics.values())
            f.write(values + "\n")
    
    def print_summary(self, last_n: int = 100) -> None:
        """
        Imprime un resumen de las últimas N métricas.
        
        Args:
            last_n: Número de últimos valores a promediar
        """
        if not self.metrics_buffer:
            print("No metrics to summarize")
            return
        
        print("\n" + "="*60)
        print(f"METRICS SUMMARY (last {last_n} steps)")
        print("="*60)
        
        for key, values in self.metrics_buffer.items():
            if key == "step":
                continue
            
            recent_values = values[-last_n:]
            mean = np.mean(recent_values)
            std = np.std(recent_values)
            min_val = np.min(recent_values)
            max_val = np.max(recent_values)
            
            print(f"{key:20s}: {mean:8.3f} ± {std:6.3f}  (min: {min_val:7.3f}, max: {max_val:7.3f})")
        
        print("="*60 + "\n")
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Retorna todas las métricas acumuladas."""
        return self.metrics_buffer


# ESTADÍSTICAS DE EPISODIOS

def compute_episode_stats(episode_rewards: List[float]) -> Dict[str, float]:
    """
    Calcula estadísticas de una lista de recompensas de episodios.
    
    Args:
        episode_rewards: Lista de recompensas totales por episodio
    
    Returns:
        Diccionario con mean, std, min, max
    """
    if not episode_rewards:
        return {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
        }
    
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calcula la varianza explicada: 1 - Var[y_true - y_pred] / Var[y_true]
    
    Esta métrica mide qué tan bien el value function predice los returns.
    - 1.0 = predicción perfecta
    - 0.0 = tan bueno como predecir la media
    - <0.0 = peor que predecir la media
    
    Args:
        y_pred: Valores predichos (e.g., values)
        y_true: Valores reales (e.g., returns)
    
    Returns:
        Explained variance (float)
    """
    assert y_pred.shape == y_true.shape, "Shapes must match"
    
    var_y = np.var(y_true)
    if var_y == 0:
        # Si la varianza es 0, no hay nada que explicar
        return np.nan
    
    return 1.0 - np.var(y_true - y_pred) / var_y

# ANNEALING Y SCHEDULING

def linear_anneal(
    step: int,
    total_steps: int,
    start_value: float,
    end_value: float
) -> float:
    """
    Annealing lineal de un valor desde start_value hasta end_value.
    
    Útil para:
    - Learning rate decay
    - Epsilon decay en exploration
    - Clip epsilon decay
    
    Args:
        step: Paso actual
        total_steps: Total de pasos
        start_value: Valor inicial
        end_value: Valor final
    
    Returns:
        Valor interpolado linealmente
    
    Example:
        >>> # Learning rate decay de 1e-3 a 1e-4 en 1M steps
        >>> lr = linear_anneal(step=500_000, total_steps=1_000_000, 
        ...                    start_value=1e-3, end_value=1e-4)
        >>> print(lr)  # 5.5e-4
    """
    if step >= total_steps:
        return end_value
    
    fraction = step / total_steps
    return start_value + fraction * (end_value - start_value)


# DEVICE MANAGEMENT

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Obtiene el device apropiado (cuda/mps/cpu).
    
    Args:
        device: Device específico ("cuda", "cpu", "mps", None)
                Si None, selecciona automáticamente el mejor disponible
    
    Returns:
        torch.device
    
    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda")  # Force CUDA
    """
    if device is not None:
        return torch.device(device)
    
    # Auto-detect mejor device disponible
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2)
        return torch.device("mps")
    else:
        return torch.device("cpu")


def move_to_device(data, device: torch.device):
    """
    Mueve datos (tensor, lista, tupla, dict) a un device.
    
    Args:
        data: Datos a mover (tensor, list, tuple, dict)
        device: Device destino
    
    Returns:
        Datos en el device especificado
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


# GRADIENT UTILITIES

def clip_grad_norm(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> float:
    """
    Wrapper sobre torch.nn.utils.clip_grad_norm_ con mejor interfaz.
    
    Clipea los gradientes para prevenir explosión.
    
    Args:
        parameters: Parámetros del modelo (model.parameters())
        max_norm: Norma máxima de los gradientes
        norm_type: Tipo de norma (2.0 = L2 norm)
        error_if_nonfinite: Si True, arroja error si hay NaN/Inf
    
    Returns:
        Total norm de los gradientes antes del clipping
    
    Example:
        >>> # En el training loop
        >>> loss.backward()
        >>> grad_norm = clip_grad_norm(model.parameters(), max_norm=0.5)
        >>> optimizer.step()
    """
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite
    )


def get_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """
    Calcula la norma de los gradientes sin clipear.
    Útil para logging/debugging.
    
    Args:
        parameters: Parámetros del modelo
        norm_type: Tipo de norma
    
    Returns:
        Norma total de los gradientes
    """
    parameters = [p for p in parameters if p.grad is not None]
    
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
    
    return total_norm.item()


# CHECKPOINTS (GUARDAR/CARGAR MODELOS)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    save_dir: str,
    filename: Optional[str] = None,
    config: Optional[Dict] = None,
) -> str:
    """
    Guarda un checkpoint del modelo y optimizer.
    
    Args:
        model: Modelo de PyTorch
        optimizer: Optimizer de PyTorch
        step: Paso de entrenamiento actual
        save_dir: Directorio donde guardar
        filename: Nombre del archivo (opcional, se genera automáticamente)
        config: Configuración del modelo (opcional)
    
    Returns:
        Path del archivo guardado
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_step_{step}.pt"
    
    filepath = save_dir / filename
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")
    
    return str(filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    """
    Carga un checkpoint del modelo y optimizer.
    
    Args:
        filepath: Path al archivo de checkpoint
        model: Modelo donde cargar los pesos
        optimizer: Optimizer donde cargar el estado (opcional)
        device: Device donde cargar el modelo
    
    Returns:
        Paso de entrenamiento del checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    step = checkpoint.get("step", 0)
    
    print(f"✓ Checkpoint loaded from: {filepath} (step {step})")
    
    return step


def save_model_only(
    model: torch.nn.Module,
    save_path: str,
) -> None:
    """
    Guarda solo el modelo (sin optimizer ni metadata).
    Útil para el modelo final.
    
    Args:
        model: Modelo de PyTorch
        save_path: Path completo del archivo
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"✓ Model saved: {save_path}")


def load_model_only(
    model: torch.nn.Module,
    load_path: str,
    device: str = "cpu",
) -> None:
    """
    Carga solo el modelo (sin optimizer).
    
    Args:
        model: Modelo donde cargar los pesos
        load_path: Path al archivo del modelo
        device: Device donde cargar
    """
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Model loaded from: {load_path}")


# CONFIGURACIÓN Y METADATA

def save_config(config_dict: Dict, save_path: str) -> None:
    """
    Guarda la configuración en formato JSON.
    
    Args:
        config_dict: Diccionario de configuración
        save_path: Path donde guardar
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✓ Config saved: {save_path}")


def load_config(load_path: str) -> Dict:
    """
    Carga configuración desde JSON.
    
    Args:
        load_path: Path al archivo de configuración
    
    Returns:
        Diccionario de configuración
    """
    with open(load_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Config loaded from: {load_path}")
    return config


# INFORMACIÓN DEL SISTEMA

def print_system_info() -> None:
    """Imprime información del sistema (GPU, CUDA, etc.)."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"CUDA available: No (using CPU)")
    
    print("="*60 + "\n")

# TESTING

def test_utils():
    """Test de las funciones de utils."""
    print("Testing utils.py...")
    
    # Test set_seed
    print("\n1. Testing set_seed:")
    set_seed(42)
    val1 = np.random.rand()
    set_seed(42)
    val2 = np.random.rand()
    assert val1 == val2, "Seeds not working properly"
    print("   ✓ set_seed works")
    
    # Test Logger
    print("\n2. Testing Logger:")
    logger = Logger("/tmp/test_logs", "test")
    logger.log({"reward": 10.5, "loss": 0.3}, step=1)
    logger.log({"reward": 12.0, "loss": 0.25}, step=2)
    logger.print_summary(last_n=2)
    print("   ✓ Logger works")
    
    # Test compute_episode_stats
    print("\n3. Testing compute_episode_stats:")
    rewards = [10.0, 15.0, 12.0, 20.0]
    stats = compute_episode_stats(rewards)
    print(f"   Stats: {stats}")
    assert "mean_reward" in stats
    print("   ✓ compute_episode_stats works")
    
    # Test compute_explained_variance
    print("\n4. Testing compute_explained_variance:")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    explained_var = compute_explained_variance(y_pred, y_true)
    print(f"   Explained variance: {explained_var:.4f}")
    assert 0.0 <= explained_var <= 1.0
    print("   ✓ compute_explained_variance works")
    
    # Test linear_anneal
    print("\n5. Testing linear_anneal:")
    lr_start = linear_anneal(0, 1000, 1e-3, 1e-4)
    lr_mid = linear_anneal(500, 1000, 1e-3, 1e-4)
    lr_end = linear_anneal(1000, 1000, 1e-3, 1e-4)
    print(f"   Start: {lr_start:.6f}, Mid: {lr_mid:.6f}, End: {lr_end:.6f}")
    assert lr_start == 1e-3
    assert lr_end == 1e-4
    assert lr_start > lr_mid > lr_end
    print("   ✓ linear_anneal works")
    
    # Test get_device
    print("\n6. Testing get_device:")
    device = get_device()
    print(f"   Auto-detected device: {device}")
    device_cpu = get_device("cpu")
    assert device_cpu.type == "cpu"
    print("   ✓ get_device works")
    
    # Test gradient utilities
    print("\n7. Testing gradient utilities:")
    dummy_model = torch.nn.Linear(10, 5)
    dummy_input = torch.randn(2, 10)
    dummy_output = dummy_model(dummy_input)
    loss = dummy_output.sum()
    loss.backward()
    
    grad_norm_before = get_grad_norm(dummy_model.parameters())
    grad_norm_clipped = clip_grad_norm(dummy_model.parameters(), max_norm=1.0)
    print(f"   Grad norm before clip: {grad_norm_before:.4f}")
    print(f"   Grad norm after clip: {grad_norm_clipped:.4f}")
    print("   ✓ Gradient utilities work")
    
    # Test checkpoints
    print("\n8. Testing checkpoints:")
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    
    save_checkpoint(dummy_model, dummy_optimizer, step=100, save_dir="/tmp/test_checkpoints")
    step = load_checkpoint("/tmp/test_checkpoints/checkpoint_step_100.pt", dummy_model, dummy_optimizer)
    assert step == 100
    print("   ✓ Checkpoints work")
    
    # Test save/load model only
    print("\n9. Testing model save/load:")
    save_model_only(dummy_model, "/tmp/test_model.pt")
    load_model_only(dummy_model, "/tmp/test_model.pt")
    print("   ✓ Model save/load works")
    
    # Test config
    print("\n10. Testing config save/load:")
    test_config = {"lr": 0.001, "gamma": 0.99}
    save_config(test_config, "/tmp/test_config.json")
    loaded_config = load_config("/tmp/test_config.json")
    assert loaded_config == test_config
    print("   ✓ Config save/load works")
    
    # System info
    print("\n11. Testing system info:")
    print_system_info()
    
    print("\n" + "="*60)
    print("✓ All utils tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_utils()