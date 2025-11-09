"""
Rollout buffer para PPO.

Almacena las transiciones recolectadas durante la fase de rollout.
"""
import torch
import numpy as np
from typing import Optional, Tuple


class RolloutBuffer:
    """
    Buffer para almacenar experiencias durante la fase de rollout.
    
    Almacena:
    - observations
    - actions
    - log_probs (de la política anterior)
    - rewards
    - dones (terminated/truncated)
    - values (predicciones del critic)
    
    Después de la fase de rollout, calculamos las advantages y returns usando GAE.
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        action_dim: int,
        device: str = "cpu"
    ):
        """
        Args:
            buffer_size: número de pasos a almacenar (n_steps * n_envs)
            obs_shape: forma de una observación
            action_dim: dimensión del espacio de acciones
            device: 'cpu' o 'cuda'
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        
        # Buffers
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_dim) if action_dim > 1 else (buffer_size,), dtype=torch.long)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        
        # Para GAE
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32)
        
        self.pos = 0
        self.full = False
    
    def reset(self):
        """Resetear el buffer."""
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float
    ):
        """
        Añadir una transición al buffer.
        
        Args:
            obs: observación
            action: acción tomada
            log_prob: log π_old(a|s)
            reward: recompensa recibida
            done: si el episodio terminó
            value: V(s) predicho
        """
        if self.pos >= self.buffer_size:
            raise RuntimeError("Buffer lleno! Llamar a reset() después de compute_returns_and_advantage()")
        
        # Convertir a tensores
        self.observations[self.pos] = torch.as_tensor(obs, dtype=torch.float32)
        
        if isinstance(action, (int, np.integer)):
            self.actions[self.pos] = action
        else:
            self.actions[self.pos] = torch.as_tensor(action, dtype=torch.float32 if self.action_dim > 1 else torch.long)
        
        self.log_probs[self.pos] = log_prob #pi_old(a|s)
        self.rewards[self.pos] = reward 
        self.dones[self.pos] = float(done) 
        self.values[self.pos] = value #V(s) del critic
        
        self.pos += 1
        
        if self.pos == self.buffer_size:
            self.full = True

        #aca guarde tood lo necesario para despues calcular las ventajas y los retornos
    
    def compute_returns_and_advantage(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Calcular advantages y returns usando GAE.
        
        Esta es la implementación principal usada durante el entrenamiento PPO.
        Está optimizada para trabajar directamente con los tensores del buffer
        y maneja correctamente el bootstrap con last_value del último estado.
        
        Para análisis independientes o tests, ver ppoAgent.gae.compute_gae()
        
        Args:
            last_value: V(s_T) - valor del último estado (para bootstrap)
            last_done: si el último estado es terminal
            gamma: factor de descuento
            gae_lambda: parámetro λ de GAE (trade-off bias-varianza)
        """
        # Implementación recursiva de GAE 
        last_gae_lam = 0.0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            
            # δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # A_t = δ_t + (γ*λ)*A_{t+1}
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        
        # Returns = advantages + values
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: Optional[int] = None):
        """
        Generar batches para el entrenamiento.
        
        Args:
            batch_size: tamaño del batch. Si None, devuelve todo.
        
        Yields:
            dict con keys: observations, actions, log_probs, advantages, returns, values
        """
        if not self.full:
            raise RuntimeError("Buffer no está lleno! Llamar después de llenar el buffer.")
        
        indices = np.arange(self.buffer_size)
        
        if batch_size is None:
            # Devolver todo de una vez
            yield self._get_samples(indices)
        else:
            # mezclamos y creamos mini-batches
            np.random.shuffle(indices)
            
            start_idx = 0
            while start_idx < self.buffer_size:
                batch_indices = indices[start_idx : start_idx + batch_size]
                yield self._get_samples(batch_indices)
                start_idx += batch_size
    
    def _get_samples(self, indices: np.ndarray):
        """Extraemos muestras dadas los índices."""
        # Normalizamos las advantages (truco de implementación)
        advantages = self.advantages[indices]
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        # Solo normalizar si std > 0 (evita NaN cuando todas las advantages son iguales)
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std
        else:
            # Si todas las advantages son iguales, dejarlas en cero
            advantages = advantages - adv_mean
        
        return {
            "observations": self.observations[indices].to(self.device),
            "actions": self.actions[indices].to(self.device),
            "log_probs": self.log_probs[indices].to(self.device),
            "advantages": advantages.to(self.device),
            "returns": self.returns[indices].to(self.device),
            "values": self.values[indices].to(self.device),
        }
