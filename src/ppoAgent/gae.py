"""
Generalized Advantage Estimation (GAE).

GAE combina estimadores de ventaja con diferentes trade-offs entre bias y varianza:
- λ=0: solo TD (bajo bias, alta varianza)
- λ=1: Monte Carlo (alto bias, baja varianza)

Esta función es una implementación genérica de GAE. Para el entrenamiento PPO,
se usa RolloutBuffer.compute_returns_and_advantage() que está optimizada para el buffer
y maneja correctamente el bootstrap con last_value.

Esta función es útil para:
- Tests unitarios
- Análisis independientes de GAE
- Múltiples entornos paralelos (usar compute_gae_vectorized)
"""
import torch
import numpy as np


def compute_gae(
    rewards,
    values,
    dones,
    gamma=0.99,
    gae_lambda=0.95,
    use_proper_time_limits=True
):
    """
    Calcular GAE (Generalized Advantage Estimation).
    
    Implementación recursiva como en las notas de clase:
    
    δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
    A_t = δ_t + (γ*λ)*δ_{t+1} + (γ*λ)²*δ_{t+2} + ...
    
    Args:
        rewards: array [T] o [T, N] de rewards
        values: array [T] o [T, N] de valores V(s_t) 
        dones: array [T] o [T, N] de flags de terminación
        gamma: factor de descuento
        gae_lambda: parámetro λ para GAE
        use_proper_time_limits: si True, no propaga valores después de truncation
    
    Returns:
        advantages: array [T] o [T, N] de ventajas
        returns: array [T] o [T, N] de retornos (targets para el critic)
    """
    # Convertir a numpy si es necesario
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.cpu().numpy()
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    if isinstance(dones, torch.Tensor):
        dones = dones.cpu().numpy()
    
    # Asegurar que sean arrays 1D o 2D
    rewards = np.asarray(rewards)
    values = np.asarray(values)
    dones = np.asarray(dones).astype(np.float32)
    
    T = len(rewards)
    advantages = np.zeros_like(rewards)
    
    # Computar hacia atrás (más eficiente)
    last_gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            # Último paso: no hay next_value
            next_value = 0.0
            next_not_done = 0.0
        else:
            next_value = values[t + 1]
            next_not_done = 1.0 - dones[t]
        
        # δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * next_not_done - values[t]
        
        # A_t = δ_t + (γ*λ)*A_{t+1}
        advantages[t] = delta + gamma * gae_lambda * next_not_done * last_gae
        last_gae = advantages[t]
    
    # Returns = A + V (target para el critic)
    returns = advantages + values
    
    return advantages, returns


def compute_gae_vectorized(
    rewards,
    values, 
    dones,
    gamma=0.99,
    gae_lambda=0.95
):
    """
    Versión vectorizada de GAE para batches de episodios.
    
    Args:
        rewards: tensor [T, N] donde N es num de envs paralelos
        values: tensor [T, N]
        dones: tensor [T, N]
        gamma: float
        gae_lambda: float
    
    Returns:
        advantages: tensor [T, N]
        returns: tensor [T, N]
    """
    device = rewards.device if isinstance(rewards, torch.Tensor) else 'cpu'
    
    # Asegurar tensores
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float32, device=device)
    if not isinstance(dones, torch.Tensor):
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
    
    T, N = rewards.shape if rewards.ndim == 2 else (rewards.shape[0], 1)
    
    # Reshape si es 1D
    if rewards.ndim == 1:
        rewards = rewards.view(-1, 1)
        values = values.view(-1, 1)
        dones = dones.view(-1, 1)
    
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=device)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = torch.zeros(N, device=device)
            next_not_done = torch.zeros(N, device=device)
        else:
            next_value = values[t + 1]
            next_not_done = 1.0 - dones[t]
        
        delta = rewards[t] + gamma * next_value * next_not_done - values[t]
        advantages[t] = delta + gamma * gae_lambda * next_not_done * last_gae
        last_gae = advantages[t]
    
    returns = advantages + values
    
    # Reshape de vuelta si era 1D
    if N == 1:
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
    
    return advantages, returns
