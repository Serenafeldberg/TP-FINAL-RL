#!/usr/bin/env python3
"""
train the in-house PPO implementation on CartPole-v1 for 200k timesteps.

this script reuses the PPO components from ``src/ppoAgent`` and mirrors
the hyperparameters used by Stable Baselines 3 for direct comparison.

salidas:
    - ``savedModels/cartpole_custom/ppo_cartpole_custom_final.pth`` (policy)
    - ``cartpole_custom_rewards.npy`` with rewards por episodio
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config  # type: ignore  # noqa: E402
from ppoAgent.actorCritic import ActorCritic  # type: ignore  # noqa: E402
from ppoAgent.memory import RolloutBuffer  # type: ignore  # noqa: E402
from ppoAgent.ppo import PPO  # type: ignore  # noqa: E402
from ppoAgent.utils import set_seed  # type: ignore  # noqa: E402

HIDDEN_SIZE = 64  
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
TOTAL_TIMESTEPS = 200_000
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.0
MAX_GRAD_NORM = 0.5
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR_DECAY = True
SOLVE_WINDOW = 100
SOLVE_THRESHOLD = 475.0

OUTPUT_DIR = BASE_DIR / "savedModels" / "cartpole_custom"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REWARDS_PATH = BASE_DIR / "cartpole_custom_rewards.npy"
MODEL_PATH = OUTPUT_DIR / "ppo_cartpole_custom_final.pth"


def create_env(seed: int) -> gym.Env:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    return env


def log_progress(episode: int, reward: float, solved: bool) -> None:
    status = "✓ solved" if solved else ""
    print(f"Episode {episode:4d}: reward={reward:7.2f} {status}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO (custom) on CartPole-v1.")
    parser.add_argument("--seed", type=int, default=Config.SEED or 42)
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    env = create_env(args.seed)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"Obs shape: {obs_shape}, Action dim: {action_dim}")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Solve criterion: Mean{SOLVE_WINDOW} >= {SOLVE_THRESHOLD}")
    print("-" * 70)

    actor_critic = ActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        action_type="discrete",
        hidden_size=HIDDEN_SIZE,
    )
    agent = PPO(
        actor_critic=actor_critic,
        learning_rate=LEARNING_RATE,
        clip_epsilon=CLIP_EPSILON,
        value_loss_coef=VALUE_LOSS_COEF,
        entropy_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        lr_decay=LR_DECAY,
        device=args.device,
    )

    buffer = RolloutBuffer(
        buffer_size=N_STEPS,
        obs_shape=obs_shape,
        action_dim=1,
        device=args.device,
    )

    # NO RunningNorm - CartPole no necesita normalizacion de observacion, SB3 tampoco lo hace por defecto
    rewards_history: List[float] = []
    mean_window: float = 0.0
    solved_at: int | None = None

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    total_updates = TOTAL_TIMESTEPS // N_STEPS
    total_timesteps_tracked = 0
    start_time = time.time()

    for update in range(1, total_updates + 1):
        for step in range(N_STEPS):
            action, log_prob, value = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(
                obs=torch.as_tensor(obs, dtype=torch.float32, device=args.device),
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
            )

            episode_reward += reward
            obs = next_obs
            total_timesteps_tracked += 1

            if done:
                rewards_history.append(episode_reward)
                window = rewards_history[-SOLVE_WINDOW:]
                mean_window = float(np.mean(window))
                solved = len(rewards_history) >= SOLVE_WINDOW and mean_window >= SOLVE_THRESHOLD
                
                if len(rewards_history) % 10 == 0 or solved:
                    log_progress(len(rewards_history), episode_reward, solved)
                
                if solved and solved_at is None:
                    solved_at = len(rewards_history)

                obs, _ = env.reset()
                episode_reward = 0.0

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            last_value = agent.actor_critic.get_value(obs_tensor).item()

        buffer.compute_returns_and_advantage(
            last_value=last_value,
            last_done=done,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        agent.update(
            rollout_buffer=buffer,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            total_timesteps=TOTAL_TIMESTEPS,
            current_timestep=total_timesteps_tracked,
        )
        buffer.reset()

        if update % 5 == 0:
            recent_rewards = rewards_history[-min(100, len(rewards_history)):]
            mean_recent = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            print(
                f"Update {update:2d}/{total_updates} | Timesteps: {total_timesteps_tracked:6d} | "
                f"Episodes: {len(rewards_history):4d} | Mean100: {mean_recent:6.2f}"
            )

        if len(rewards_history) >= SOLVE_WINDOW and mean_window >= SOLVE_THRESHOLD:
            print(f"\n{'='*70}")
            print(f"CartPole resuelto despues de {len(rewards_history)} episodios!")
            print(f"   Mean{SOLVE_WINDOW}: {mean_window:.2f} >= {SOLVE_THRESHOLD}")
            print(f"   Total timesteps: {total_timesteps_tracked:,}")
            print(f"{'='*70}")
            break

    elapsed = time.time() - start_time
    agent.save(str(MODEL_PATH))
    np.save(REWARDS_PATH, np.asarray(rewards_history, dtype=np.float32))

    print("\n" + "="*70)
    print("RESUMEN DE ENTRENAMIENTO:")
    print("="*70)
    print(f"  Episodes:          {len(rewards_history)}")
    print(f"  Timesteps:         {total_timesteps_tracked:,}")
    print(f"  Solved at episode: {solved_at if solved_at else '-'}")
    print(f"  Mean{SOLVE_WINDOW} final:  {mean_window:.2f}")
    print(f"  Wall time:         {elapsed/60:.1f} min")
    print(f"  Rewards saved to:  {REWARDS_PATH}")
    print(f"  Model saved to:    {MODEL_PATH}")
    print("="*70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())