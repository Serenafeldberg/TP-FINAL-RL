import os
import random
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path
import json
from datetime import datetime

import torch as th

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

def linear_lr(opt, frac: float):
    """frac en [0,1] — 1 al inicio, 0 al final."""
    for pg in opt.param_groups:
        base = pg.get("initial_lr", pg["lr"])
        pg["lr"] = base * max(0.0, min(1.0, frac))

#REPRODUCIBILIDAD
def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Setea la seed para reproducibilidad en todos los frameworks.
    
    Args:
        seed: Seed para random, numpy, torch, etc.
        deterministic: Si True, activa modo determinístico completo (más lento)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
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


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
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


# Alias para compatibilidad
compute_explained_variance = explained_variance


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


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Crear un scheduler lineal para learning rate.
    
    Args:
        initial_value: valor inicial
        final_value: valor final
    
    Returns:
        función que toma progress (0 a 1) y devuelve el valor interpolado
    """
    def func(progress: float) -> float:
        """
        Args:
            progress: float en [0, 1], donde 0 = inicio, 1 = final
        """
        return final_value + (initial_value - final_value) * (1 - progress)
    
    return func


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
    Clipea el norm del gradiente de un iterable de parámetros.
    Wrapper sobre torch.nn.utils.clip_grad_norm_.
    
    Args:
        parameters: Iterable de parámetros (e.g., model.parameters())
        max_norm: Norm máximo
        norm_type: Tipo de norm (default: 2.0 = L2)
        error_if_nonfinite: Si lanzar error si hay gradientes inf/nan
    
    Returns:
        Norm total del gradiente (antes del clipping)
    """
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite
    )


def get_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """
    Calcula el norm del gradiente sin clipear.
    
    Args:
        parameters: Iterable de parámetros
        norm_type: Tipo de norm (default: 2.0 = L2)
    
    Returns:
        Norm del gradiente
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.data, norm_type) for p in parameters]),
            norm_type
        )
    
    return total_norm.item()


def zero_grad(parameters) -> None:
    """
    Zero out gradients de parámetros.
    
    Args:
        parameters: Iterable de parámetros
    """
    for param in parameters:
        if param.grad is not None:
            param.grad.zero_()


# CHECKPOINT UTILITIES

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
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"MPS available: Yes (Apple Silicon)")
    
    print("="*60 + "\n")

# NORMALIZACIÓN DE OBSERVACIONES

class RunningNorm:
    """
    Normalizador de observaciones usando estadísticas corrientes (running statistics).
    
    Implementa normalización Z-score: (x - mean) / std
    Las estadísticas se actualizan incrementalmente usando el algoritmo de Welford.
    
    Mejoras sobre la implementación básica:
    - Actualización en batch más eficiente
    - Modo de evaluación (no actualiza estadísticas)
    - Guardado/carga de estadísticas para reproducibilidad
    - Manejo robusto de casos edge (std=0, etc.)
    
    Uso recomendado:
        # Durante entrenamiento: actualizar al final de cada rollout
        norm = RunningNorm(obs_shape)
        for rollout in rollouts:
            # Recolectar observaciones sin normalizar
            obs_batch = collect_observations(...)
            # Actualizar estadísticas con todo el batch
            norm.update_batch(obs_batch)
            # Normalizar para el siguiente rollout
            obs_normalized = norm.normalize(obs_batch)
    
    Uso durante evaluación:
        norm.eval()  # No actualizar estadísticas
        obs_norm = norm.normalize(obs)
    """
    
    def __init__(
        self,
        shape: tuple,
        eps: float = 1e-8,
        clip: Optional[float] = None
    ):
        """
        Args:
            shape: Forma de las observaciones (ej: (4,) para CartPole)
            eps: Valor pequeño para evitar división por cero
            clip: Si no None, clipea valores normalizados a [-clip, clip]
        """
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.eps = eps
        self.clip = clip
        
        # Estadísticas corrientes
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)  # Inicializar con var=1 (std=1)
        self.count = eps  # Evitar división por cero
        
        # Modo de entrenamiento/evaluación
        self.training = True
    
    def update(self, x: np.ndarray) -> None:
        """
        Actualiza las estadísticas con una sola observación.
        
        Args:
            x: Observación de forma self.shape
        """
        if not self.training:
            return
        
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.shape:
            raise ValueError(f"Expected shape {self.shape}, got {x.shape}")
        
        # Algoritmo de Welford para actualización incremental
        prev_count = self.count
        self.count += 1.0

        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean

        m2 = self.var * prev_count
        m2 += delta * delta2
        self.var = m2 / self.count
    
    def update_batch(self, x: np.ndarray) -> None:
        """
        Actualiza las estadísticas con un batch de observaciones.
        Más eficiente que llamar update() múltiples veces.
        
        Args:
            x: Array de observaciones de forma (N, *self.shape) o (*self.shape,)
        """
        if not self.training:
            return
        
        x = np.asarray(x, dtype=np.float64)
        
        # Manejar caso de observación única
        if x.shape == self.shape:
            self.update(x)
            return
        
        # Verificar que las dimensiones coincidan
        if x.shape[-len(self.shape):] != self.shape:
            raise ValueError(f"Expected last dimensions to be {self.shape}, got {x.shape}")
        
        # Flatten para procesar batch
        batch_size = int(np.prod(x.shape[:-len(self.shape)]))
        x_flat = x.reshape(batch_size, *self.shape)
        
        # Calcular estadísticas del batch
        batch_mean = x_flat.mean(axis=0)
        batch_var = x_flat.var(axis=0, ddof=0)  # ddof=0 para varianza poblacional
        batch_count = batch_size
        
        # Actualizar estadísticas globales usando fórmula de combinación
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        # Nueva media
        new_mean = self.mean + delta * batch_count / tot_count
        
        # Nueva varianza (fórmula de combinación de varianzas)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normaliza observaciones usando las estadísticas corrientes.
        
        Args:
            x: Observación(es) de forma (*self.shape) o (N, *self.shape)
        
        Returns:
            Observación(es) normalizada(s) de la misma forma
        """
        x = np.asarray(x, dtype=np.float32)
        
        # Calcular std (evitar división por cero)
        std = np.sqrt(self.var) + self.eps
        
        # Normalizar
        x_norm = (x - self.mean) / std
        
        # Clipear si se especificó
        if self.clip is not None:
            x_norm = np.clip(x_norm, -self.clip, self.clip)
        
        return x_norm.astype(np.float32)
    
    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Desnormaliza observaciones normalizadas.
        Útil para visualización o análisis.
        
        Args:
            x_norm: Observación(es) normalizada(s)
        
        Returns:
            Observación(es) original(es)
        """
        x_norm = np.asarray(x_norm, dtype=np.float64)
        std = np.sqrt(self.var) + self.eps
        x = x_norm * std + self.mean
        return x.astype(np.float32)
    
    def reset(self) -> None:
        """Resetea las estadísticas a valores iniciales."""
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = self.eps
    
    def train(self) -> None:
        """Activa modo de entrenamiento (actualiza estadísticas)."""
        self.training = True
    
    def eval(self) -> None:
        """Activa modo de evaluación (no actualiza estadísticas)."""
        self.training = False
    
    def get_stats(self) -> Dict[str, np.ndarray]:
        """
        Retorna las estadísticas actuales.
        
        Returns:
            Dict con 'mean', 'std', 'count'
        """
        return {
            'mean': self.mean.copy(),
            'std': np.sqrt(self.var),
            'count': self.count
        }
    
    def set_stats(self, mean: np.ndarray, var: np.ndarray, count: float) -> None:
        """
        Establece las estadísticas manualmente.
        Útil para cargar estadísticas guardadas.
        
        Args:
            mean: Media
            var: Varianza
            count: Número de muestras
        """
        mean = np.asarray(mean, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)
        
        if mean.shape != self.shape or var.shape != self.shape:
            raise ValueError(f"Stats shape mismatch: mean={mean.shape}, var={var.shape}, expected={self.shape}")
        
        self.mean = mean.copy()
        self.var = var.copy()
        self.count = float(count)
    
    def save(self, path: str) -> None:
        """
        Guarda las estadísticas en un archivo numpy.
        
        Args:
            path: Path donde guardar (.npz)
        """
        np.savez(
            path,
            mean=self.mean,
            var=self.var,
            count=self.count,
            shape=self.shape,
            eps=self.eps,
            clip=self.clip
        )
    
    def load(self, path: str) -> None:
        """
        Carga estadísticas desde un archivo numpy.
        
        Args:
            path: Path al archivo .npz
        """
        data = np.load(path)
        self.mean = data['mean']
        self.var = data['var']
        self.count = float(data['count'])
        
        # Verificar que la forma coincida
        loaded_shape = tuple(data['shape'])
        if loaded_shape != self.shape:
            raise ValueError(f"Shape mismatch: loaded {loaded_shape}, expected {self.shape}")
        
        if 'eps' in data:
            self.eps = float(data['eps'])
        if 'clip' in data:
            self.clip = float(data['clip']) if not np.isnan(data['clip']) else None
    
    def __repr__(self) -> str:
        mode = "train" if self.training else "eval"
        return f"RunningNorm(shape={self.shape}, count={self.count:.0f}, mode={mode})"


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
    
    # Test explained_variance
    print("\n4. Testing explained_variance:")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    explained_var = explained_variance(y_pred, y_true)
    print(f"   Explained variance: {explained_var:.4f}")
    assert 0.0 <= explained_var <= 1.0
    print("   ✓ explained_variance works")
    
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
    
    # Test RunningNorm
    print("\n12. Testing RunningNorm:")
    norm = RunningNorm(shape=(4,))
    
    # Test update single
    obs1 = np.array([1.0, 2.0, 3.0, 4.0])
    norm.update(obs1)
    assert norm.count > 0, "Count should increase"
    print("   ✓ Single update works")
    
    # Test update batch
    obs_batch = np.random.randn(10, 4).astype(np.float32)
    norm.update_batch(obs_batch)
    assert norm.count > 10, "Count should increase with batch"
    print("   ✓ Batch update works")
    
    # Test normalize
    obs_norm = norm.normalize(obs1)
    assert obs_norm.shape == obs1.shape, "Normalized shape should match"
    assert np.abs(obs_norm.mean()) < 1.0, "Normalized should be roughly zero-mean"
    print("   ✓ Normalize works")
    
    # Test denormalize
    obs_denorm = norm.denormalize(obs_norm)
    assert np.allclose(obs_denorm, obs1, atol=1e-5), "Denormalize should recover original"
    print("   ✓ Denormalize works")
    
    # Test eval mode
    norm.eval()
    old_mean = norm.mean.copy()
    norm.update(obs1)
    assert np.allclose(norm.mean, old_mean), "Eval mode should not update stats"
    print("   ✓ Eval mode works")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    
    norm.train()  # Back to training mode
    norm.save(temp_path)
    norm2 = RunningNorm(shape=(4,))
    norm2.load(temp_path)
    assert np.allclose(norm.mean, norm2.mean), "Loaded mean should match"
    assert np.allclose(norm.var, norm2.var), "Loaded var should match"
    print("   ✓ Save/load works")
    
    import os
    os.unlink(temp_path)
    
    print("   ✓ RunningNorm tests passed")
    
    print("\n" + "="*60)
    print("✓ All utils tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_utils()