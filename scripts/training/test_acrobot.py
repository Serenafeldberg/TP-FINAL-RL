#!/usr/bin/env python3
"""
quick probe run: train the custom PPO agent on Acrobot-v1 for 100k timesteps
to verify generalization beyond Flappy Bird. produce a learning curve plot and
print whether the policy reaches the defined convergence threshold.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import gymnasium as gym
import matplotlib.pyplot as plt
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
from ppoAgent.utils import RunningNorm, set_seed  # type: ignore  # noqa: E402

ENV_ID = "Acrobot-v1"
TOTAL_TIMESTEPS = 1_000_000
LR_DECAY_STEPS = 100_000  
LR_MIN_FRAC = 0.1  
HIDDEN_SIZE = 64
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.0
MAX_GRAD_NORM = 0.5
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR_DECAY = True

ROLLING_WINDOW = 20
CONVERGENCE_THRESHOLD = -300.0  

OUTPUT_DIR = BASE_DIR / "savedModels" / "probe_acrobot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = BASE_DIR / "plots" / "probe_envs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "ppo_acrobot_probe_final.pth"
REWARDS_PATH = BASE_DIR / "acrobot_rewards.npy"
CURVE_PATH = PLOTS_DIR / "acrobot_learning_curve.png"


def plot_curve(rewards: np.ndarray, window: int, path: Path) -> None:
    if rewards.size == 0:
        return
    rolling = (
        np.convolve(rewards, np.ones(window) / window, mode="valid")
        if rewards.size >= window
        else rewards
    )
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, rewards.size + 1), rewards, alpha=0.4, label="Reward por episodio")
    plt.plot(
        np.arange(window, window + rolling.size),
        rolling,
        label=f"Media móvil ({window})",
        linewidth=2,
    )
    plt.axhline(CONVERGENCE_THRESHOLD, color="red", linestyle="--", label="Umbral convergencia")
    plt.xlabel("Episodio")
    plt.ylabel("Reward")
    plt.title("Acrobot-v1 · PPO custom (probe)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe PPO (custom) on Acrobot-v1.")
    parser.add_argument("--seed", type=int, default=Config.SEED or 123)
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    env = gym.make(ENV_ID)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

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
        lr_decay_steps=LR_DECAY_STEPS,
        lr_min_frac=LR_MIN_FRAC,
    )

    buffer = RolloutBuffer(
        buffer_size=N_STEPS,
        obs_shape=obs_shape,
        action_dim=1,
        device=args.device,
    )
    obs_norm = RunningNorm(obs_shape)

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    rewards_history: List[float] = []
    total_updates = TOTAL_TIMESTEPS // N_STEPS
    start_time = time.time()
    converged_episode: int | None = None

    for update in range(1, total_updates + 1):
        for _ in range(N_STEPS):
            obs_norm.update(obs)
            obs_normalized = obs_norm.normalize(obs)
            action, log_prob, value = agent.get_action(obs_normalized)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(
                obs=torch.as_tensor(obs_normalized, dtype=torch.float32, device=args.device),
                action=action,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=value,
            )

            episode_reward += reward
            obs = next_obs

            if done:
                rewards_history.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0.0

        with torch.no_grad():
            obs_normed = obs_norm.normalize(obs)
            obs_tensor = torch.as_tensor(obs_normed, dtype=torch.float32, device=args.device).unsqueeze(0)
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
            current_timestep=update * N_STEPS,
        )
        buffer.reset()

        if rewards_history:
            recent = rewards_history[-ROLLING_WINDOW:]
            rolling_mean = float(np.mean(recent))
            if converged_episode is None and len(rewards_history) >= ROLLING_WINDOW:
                if rolling_mean >= CONVERGENCE_THRESHOLD:
                    converged_episode = len(rewards_history)
            if update % 5 == 0:
                print(
                    f"[Update {update:02d}/{total_updates}] Episodes: {len(rewards_history):4d} | "
                    f"Mean{ROLLING_WINDOW}: {rolling_mean:7.2f}"
                )

    elapsed = time.time() - start_time
    agent.save(str(MODEL_PATH))
    obs_norm.save(OUTPUT_DIR / "obs_norm_stats_acrobot.npz")
    np.save(REWARDS_PATH, np.asarray(rewards_history, dtype=np.float32))
    plot_curve(np.asarray(rewards_history, dtype=np.float32), ROLLING_WINDOW, CURVE_PATH)

    final_mean = float(np.mean(rewards_history[-ROLLING_WINDOW:])) if rewards_history else float("nan")
    converged = final_mean >= CONVERGENCE_THRESHOLD if rewards_history else False

    print("\n=== ACROBOT PROBE SUMMARY ===")
    print(f"Episodes:           {len(rewards_history)}")
    print(f"Wall time:          {elapsed/60:.1f} min")
    print(f"Final Mean{ROLLING_WINDOW}: {final_mean:.2f}")
    print(f"Converged (>{CONVERGENCE_THRESHOLD}): {'yes' if converged else 'no'}")
    print(f"Converged episode:  {converged_episode if converged_episode else '-'}")
    print(f"Rewards saved to:   {REWARDS_PATH}")
    print(f"Curve saved to:     {CURVE_PATH}")
    print("=============================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

