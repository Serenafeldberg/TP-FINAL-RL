"""
Gym wrappers used by the training pipeline for Flappy Bird.

Includes:
- RewardClip: clip rewards to [-1, 1] (opcional).
- TimeLimit: simple step-limited wrapper.
- make_env: helper to build an env with common wrappers for PPO.

Nota: No incluye wrappers específicos de imágenes (FrameStack, PreprocessObs, etc.)
      ya que Flappy Bird usa observaciones vectoriales (LIDAR).
"""
from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore


# --- Helpers ----------------------------------------------------------------

def _unpack_step(result):
    """Handle both Gym and Gymnasium step() signatures."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info
    else:
        raise RuntimeError("Unexpected env.step() return signature")


# --- Wrappers ---------------------------------------------------------------

class RewardClip(gym.RewardWrapper):
    """Clip rewards into fixed [low, high] range."""
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward):
        return float(np.clip(reward, self.low, self.high))


class TimeLimit(gym.Wrapper):
    """End episode after max_episode_steps timesteps."""
    def __init__(self, env, max_episode_steps: int):
        super().__init__(env)
        assert max_episode_steps > 0
        self._max_episode_steps = int(max_episode_steps)
        self._elapsed_steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = _unpack_step(self.env.step(action))
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class StepAPICompat(gym.Wrapper):
    """
    Normaliza la API:
      - step(): (obs, reward, done, info) -> (obs, reward, terminated, truncated, info)
      - reset(): obs -> (obs, {})
    Útil cuando el env subyacente usa la firma 'vieja'.
    """
    def step(self, action):
        out = self.env.step(action)
        # Firma vieja: 4 elementos
        if isinstance(out, (tuple, list)) and len(out) == 4:
            obs, reward, done, info = out
            # Si el corte fue por límite de tiempo, muchos envs ponen un flag en info:
            truncated = bool(info.get("TimeLimit.truncated", False))
            # Si fue truncado por tiempo, no es 'terminated' por condición terminal del entorno:
            terminated = bool(done) and not truncated
            return obs, reward, terminated, truncated, info
        # Firma nueva: ya son 5
        return out

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        # Firma vieja: sólo obs
        if not (isinstance(res, (tuple, list)) and len(res) == 2):
            return res, {}
        return res


class NormalizeObs(gym.ObservationWrapper):
    """
    Normaliza observaciones vectoriales a [0, 1] o [-1, 1].
    Útil para estabilizar el entrenamiento con observaciones LIDAR.
    """
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high
        
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Box):
            # Actualizar observation_space para reflejar la normalización
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                shape=obs_space.shape,
                dtype=np.float32
            )
    
    def observation(self, obs):
        """Normaliza la observación al rango [low, high]."""
        obs = np.asarray(obs, dtype=np.float32)
        
        # Normalizar basándose en los límites del espacio original
        obs_space = self.env.observation_space
        if isinstance(obs_space, spaces.Box):
            original_low = obs_space.low
            original_high = obs_space.high
            
            # Normalizar a [0, 1] primero
            obs_normalized = (obs - original_low) / (original_high - original_low + 1e-8)
            
            # Escalar a [low, high]
            obs_scaled = obs_normalized * (self.high - self.low) + self.low
            
            return np.clip(obs_scaled, self.low, self.high)
        
        return obs


# --- Factory ----------------------------------------------------------------

def make_env(
    env_id: str,
    seed: Optional[int] = None,
    clip_rewards: bool = False,
    max_episode_steps: Optional[int] = None,
    normalize_obs: bool = False,
    **kwargs,
):
    """
    Create an environment and chain common wrappers for PPO training.
    
    Args:
        env_id: ID del entorno Gymnasium
        seed: Seed para reproducibilidad
        clip_rewards: Si clipear rewards a [-1, 1]
        max_episode_steps: Límite máximo de pasos por episodio (None = sin límite)
        normalize_obs: Si normalizar observaciones vectoriales
        **kwargs: Argumentos adicionales para gym.make()
    
    Returns:
        Entorno envuelto con wrappers apropiados
    """
    env = gym.make(env_id, **kwargs)

    # Compatibilidad de API (Gym vs Gymnasium)
    env = StepAPICompat(env)

    # Seed
    if seed is not None:
        try:
            env.reset(seed=seed)
        except Exception:
            try:
                env.seed(seed)
            except Exception:
                pass
        for space_name in ("action_space", "observation_space"):
            try:
                getattr(env, space_name).seed(seed)
            except Exception:
                pass

    # Time limit
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)

    # Normalización de observaciones (opcional)
    if normalize_obs:
        env = NormalizeObs(env)

    # Reward clipping (opcional)
    if clip_rewards:
        env = RewardClip(env)

    return env
