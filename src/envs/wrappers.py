"""
Gym wrappers used by the training pipeline.

Includes:
- ActionRepeat: repeat an action several frames.
- FrameStack: stack last k observations (channels-first output).
- RewardClip: clip rewards to [-1, 1].
- TimeLimit: simple step-limited wrapper.
- PreprocessObs: optional observation preprocessing (gray/resize/normalize).
- make_env: helper to build an env with common wrappers for PPO.
"""
from __future__ import annotations

import collections
from typing import Deque, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore

from . import preprocess


# --- Helpers ----------------------------------------------------------------

def _unpack_step(result):
    """Handle both Gym and Gymnasium step() signatures."""
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info
    else:
        raise RuntimeError("Unexpected env.step() return signature")


# --- Wrappers ---------------------------------------------------------------

class ActionRepeat(gym.Wrapper):
    """Repeat same action for N steps and sum rewards."""
    def __init__(self, env, repeat: int = 1):
        super().__init__(env)
        assert repeat >= 1
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        info = {}
        for _ in range(self.repeat):
            result = self.env.step(action)
            obs, reward, terminated, truncated, info = _unpack_step(result)
            total_reward += reward
            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info
        return obs, total_reward, terminated, truncated, info


class FrameStack(gym.ObservationWrapper):
    """Stack the last k observations, output in CHW format."""
    def __init__(self, env, k: int):
        super().__init__(env)
        self.k = k
        self.frames: Deque[np.ndarray] = collections.deque(maxlen=k)

        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box), "FrameStack expects Box space"
        shp = obs_space.shape

        if obs_space.dtype == np.uint8:
            low_v, high_v, dtype = 0, 255, np.uint8
        else:
            low_v, high_v, dtype = float(obs_space.low.min()), float(obs_space.high.max()), np.float32

        if len(shp) == 3:  # HWC
            H, W, C = shp
            new_shape = (C * k, H, W)
        elif len(shp) == 2:
            H, W = shp
            new_shape = (k, H, W)
        else:
            raise ValueError(f"Unsupported observation shape: {shp}")

        self.observation_space = spaces.Box(
            low=low_v, high=high_v, shape=new_shape, dtype=dtype
        )

    def _to_chw(self, obs):
        arr = np.asarray(obs)
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 2:
            arr = arr[None, ...]
        return arr

    def observation(self, obs):
        arr = self._to_chw(obs)
        self.frames.append(arr)
        while len(self.frames) < self.k:
            self.frames.appendleft(arr.copy())
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs, info = (res if isinstance(res, tuple) else (res, {}))
        self.frames.clear()
        arr = self._to_chw(obs)
        for _ in range(self.k):
            self.frames.append(arr.copy())
        stacked = np.concatenate(list(self.frames), axis=0)
        return (stacked, info) if info else stacked


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
            done = True
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
    Útil cuando el env subyacente usa la firma 'vieja' (p.ej., ALE).
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


class PreprocessObs(gym.ObservationWrapper):
    """Apply grayscale, resize and normalization to image observations."""
    def __init__(
        self,
        env,
        gray: bool = False,
        resize_shape: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
    ):
        super().__init__(env)
        self.gray = gray
        self.resize_shape = resize_shape
        self.normalize = normalize

        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Box):
            if self.gray:
                H, W = obs_space.shape[:2]
                if self.resize_shape:
                    H, W = self.resize_shape
                if self.normalize:
                    self.observation_space = spaces.Box(
                        low=0.0, high=1.0, shape=(H, W), dtype=np.float32
                    )
                else:
                    self.observation_space = spaces.Box(
                        low=0, high=255, shape=(H, W), dtype=np.uint8
                    )
            else:
                H, W, C = obs_space.shape if len(obs_space.shape) == 3 else (*obs_space.shape, 1)
                if self.resize_shape:
                    H, W = self.resize_shape
                if self.normalize:
                    self.observation_space = spaces.Box(
                        low=0.0, high=1.0, shape=(H, W, C), dtype=np.float32
                    )
                else:
                    self.observation_space = spaces.Box(
                        low=0, high=255, shape=(H, W, C), dtype=np.uint8
                    )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        arr = obs
        if self.gray:
            arr = preprocess.to_gray(arr)
        if self.resize_shape is not None:
            arr = preprocess.resize(arr, self.resize_shape)
        if self.normalize:
            arr = preprocess.normalize_obs(arr)
        return arr


# --- Factory ----------------------------------------------------------------

def make_env(
    env_id: str,
    seed: Optional[int] = None,
    frame_stack: int = 4,
    action_repeat: int = 1,
    clip_rewards: bool = True,
    max_episode_steps: Optional[int] = None,
    gray: bool = False,
    resize_shape: Optional[Tuple[int, int]] = None,
    normalize_obs: bool = True,
    **kwargs,
):
    """Create an environment and chain common wrappers for PPO training."""
    env = gym.make(env_id, **kwargs)

    env = StepAPICompat(env)

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

    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)

    if action_repeat and action_repeat > 1:
        env = ActionRepeat(env, repeat=action_repeat)

    if gray or resize_shape or normalize_obs:
        env = PreprocessObs(
            env, gray=gray, resize_shape=resize_shape, normalize=normalize_obs
        )

    if frame_stack and frame_stack > 1:
        env = FrameStack(env, k=frame_stack)

    if clip_rewards:
        env = RewardClip(env)

    return env
