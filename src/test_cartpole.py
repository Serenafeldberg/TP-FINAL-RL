"""
Entrenamiento de PPO en CartPole-v1 usando tu implementación actual.
Objetivo: verificar que el algoritmo aprende correctamente en un entorno sencillo.
"""

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from ppoAgent.utils import set_seed
from ppoAgent.actorCritic import ActorCritic
from ppoAgent.ppo import PPO
from ppoAgent.memory import RolloutBuffer

# =====================================================
# Normalizador simple de observaciones (running mean/std)
# =====================================================
class RunningNorm:
    def __init__(self, shape, eps=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.array(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = 1
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# =====================================================
# Script principal
# =====================================================
def train_cartpole():
    print("=" * 60)
    print("ENTRENAMIENTO PPO EN CARTPOLE-V1")
    print("=" * 60)

    # Config inicial
    env = gym.make("CartPole-v1")
    device = torch.device("cpu")
    set_seed(42)

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    print(f"Obs shape: {obs_shape}, Actions: {action_dim}")

    # Normalizador de observaciones
    obs_norm = RunningNorm(obs_shape)

    # Crear modelo y agente PPO
    model = ActorCritic(obs_shape, action_dim, action_type="discrete").to(device)
    agent = PPO(
        actor_critic=model,
        learning_rate=1e-3,       # más alto para CartPole
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.0,         # sin incentivo a explorar
        max_grad_norm=0.5,
        lr_decay=True,
        device=device,
    )

    # Buffer y parámetros de entrenamiento
    n_steps = 1024
    batch_size = 64
    n_epochs = 10
    total_updates = 60
    gamma = 0.99
    gae_lambda = 0.95

    buffer = RolloutBuffer(
        buffer_size=n_steps,
        obs_shape=obs_shape,
        action_dim=1,
        device=device
    )

    # Entrenamiento
    obs, _ = env.reset(seed=42)
    obs_norm.update(obs)
    obs = obs_norm.normalize(obs)

    episode_rewards = []
    episode_lengths = []
    rewards_per_update = []

    current_return, current_len = 0, 0

    for update in range(1, total_updates + 1):
        for step in range(n_steps):
            with torch.no_grad():
                action, logp, entropy, value = model.get_action_and_value(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                )

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            buffer.add(
                obs=torch.as_tensor(obs, dtype=torch.float32, device=device),
                action=action,
                log_prob=logp,
                reward=torch.tensor([reward], dtype=torch.float32, device=device),
                done=torch.tensor([done], dtype=torch.float32, device=device),
                value=value
            )

            current_return += reward
            current_len += 1

            obs = next_obs
            obs_norm.update(obs)
            obs = obs_norm.normalize(obs)

            if done:
                episode_rewards.append(current_return)
                episode_lengths.append(current_len)
                current_return, current_len = 0, 0
                obs, _ = env.reset()
                obs_norm.update(obs)
                obs = obs_norm.normalize(obs)

        # bootstrap
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, last_v = model.get_action_and_value(obs_tensor)

        buffer.compute_returns_and_advantage(
            last_value=last_v.item(),
            last_done=done,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

        metrics = agent.update(
            rollout_buffer=buffer,
            batch_size=batch_size,
            n_epochs=n_epochs,
            total_timesteps=None,
            current_timestep=None
        )

        buffer.reset()

        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        rewards_per_update.append(avg_reward)

        print(f"Update {update:03d}/{total_updates} | AvgR: {avg_reward:6.1f} "
              f"| PolLoss: {metrics['policy_loss']:.4f} | ValLoss: {metrics['value_loss']:.4f} "
              f"| Ent: {-metrics['entropy_loss']:.4f} | ClipFrac: {metrics['clip_fraction']:.3f}")

        if avg_reward >= 195:
            print("\n🎯 ¡CartPole resuelto! PPO funciona correctamente.\n")
            break

    env.close()

    # Graficar curva de recompensa
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_per_update, label="Recompensa promedio (últimos 10 episodios)")
    plt.axhline(195, color="r", linestyle="--", label="Umbral resuelto")
    plt.xlabel("Update")
    plt.ylabel("Recompensa promedio")
    plt.title("Curva de aprendizaje PPO - CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_cartpole()