#!/usr/bin/env python3
"""
train Stable-Baselines3 PPO on CartPole-v1 with the same hyperparameters
used in ``train_cartpole_custom.py`` for a clean side-by-side comparison.

salidas:
    - ``savedModels/cartpole_sb3/ppo_cartpole_sb3_final.zip`` (SB3 checkpoint)
    - ``cartpole_sb3_rewards.npy`` with rewards por episodio
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np
from stable_baselines3 import PPO  # pyright: ignore[reportMissingImports]
from stable_baselines3.common.callbacks import BaseCallback  # pyright: ignore[reportMissingImports]
from stable_baselines3.common.vec_env import DummyVecEnv  # pyright: ignore[reportMissingImports]

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "savedModels" / "cartpole_sb3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "ppo_cartpole_sb3_final"
REWARDS_PATH = BASE_DIR / "cartpole_sb3_rewards.npy"

TOTAL_TIMESTEPS = 200_000
HYPERPARAMS = dict(
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)
SOLVE_WINDOW = 100
SOLVE_THRESHOLD = 475.0


class RewardLogger(BaseCallback):
    """callback to store episode rewards for later analysis."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.solve_episode: int | None = None

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[-1]
        if "episode" in info:
            self.episode_rewards.append(info["episode"]["r"])
            if self.solve_episode is None and len(self.episode_rewards) >= SOLVE_WINDOW:
                window = self.episode_rewards[-SOLVE_WINDOW:]
                mean_window = float(np.mean(window))
                if mean_window >= SOLVE_THRESHOLD:
                    self.solve_episode = len(self.episode_rewards)
                    print(
                        f"Solve criterion reached at episode {self.solve_episode} "
                        f"(Mean{SOLVE_WINDOW}={mean_window:.2f})"
                    )
        return True


def make_env(seed: int):
    def _init():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        env = RecordEpisodeStatistics(env)
        return env

    return _init


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train SB3 PPO on CartPole-v1.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    env = DummyVecEnv([make_env(args.seed)])

    callback = RewardLogger()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=args.seed,
        verbose=1,
        **HYPERPARAMS,
    )

    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    elapsed = time.time() - start_time

    model.save(str(MODEL_PATH))
    np.save(REWARDS_PATH, np.asarray(callback.episode_rewards, dtype=np.float32))

    print("\nSB3 training summary:")
    print(f"  Episodes: {len(callback.episode_rewards)}")
    if callback.episode_rewards:
        window = callback.episode_rewards[-SOLVE_WINDOW:]
        mean_window = float(np.mean(window))
        solved = len(callback.episode_rewards) >= SOLVE_WINDOW and mean_window >= SOLVE_THRESHOLD
        print(f"  Mean{SOLVE_WINDOW} final: {mean_window:.2f}")
        print(f"  Solve criterion met: {'yes' if solved else 'no'}")
        solved_at = callback.solve_episode if callback.solve_episode else "-"
        print(f"  Solved at episode: {solved_at}")
    print(f"  Wall time: {elapsed/60:.1f} min")
    print(f"  Rewards saved to: {REWARDS_PATH}")
    print(f"  Model saved to:   {MODEL_PATH}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

