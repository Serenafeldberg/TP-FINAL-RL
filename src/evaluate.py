"""
Script para evaluar modelos PPO entrenados.

Uso:
    python evaluate.py --model path/to/model.pth --episodes 10 --render
"""
import sys
from pathlib import Path
import argparse

# Agregar src al path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
from config import Config
from envs.wrappers import make_env
from ppoAgent.actorCritic import ActorCritic
from ppoAgent.ppo import PPO


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True
):
    """
    Evaluar un modelo PPO guardado.
    
    Args:
        model_path: path al checkpoint .pth
        n_episodes: número de episodios a evaluar
        render: si renderizar el environment
        deterministic: si usar acciones determinísticas (sin sampling)
    """
    print("=" * 60)
    print("EVALUACIÓN DE MODELO PPO")
    print("=" * 60)
    print(f"Modelo: {model_path}")
    print(f"Episodios: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print("=" * 60)
    
    # Crear environment
    print("\n[1/3] Creando environment...")
    env_kwargs = Config.get_env_args()
    if render:
        env_kwargs['render_mode'] = 'human'
    
    env = make_env(
        env_id=Config.ENV_NAME,
        seed=Config.SEED,
        **env_kwargs
    )
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    action_type = "discrete" if hasattr(env.action_space, 'n') else "continuous"
    
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action dim: {action_dim}")
    
    # Crear modelo
    print("\n[2/3] Cargando modelo...")
    actor_critic = ActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        action_type=action_type,
        cnn_channels=Config.CNN_CHANNELS,
        cnn_kernels=Config.CNN_KERNELS,
        cnn_strides=Config.CNN_STRIDES,
        hidden_size=Config.HIDDEN_SIZE
    )
    
    agent = PPO(
        actor_critic=actor_critic,
        device=Config.DEVICE
    )
    
    agent.load(model_path)
    agent.actor_critic.eval()  # modo evaluación
    
    # Evaluar
    print("\n[3/3] Evaluando...")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # Seleccionar acción
            action, _, _ = agent.get_action(obs, deterministic=deterministic)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(
            f"Episode {episode + 1:2d}/{n_episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Length: {episode_length:4d}"
        )
    
    # Estadísticas
    print("-" * 60)
    print("\nESTADÍSTICAS:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min Reward:  {np.min(episode_rewards):.2f}")
    print(f"  Max Reward:  {np.max(episode_rewards):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print("=" * 60)
    
    env.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo PPO')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path al modelo .pth'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Número de episodios a evaluar (default: 10)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Renderizar el environment'
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Usar acciones estocásticas (default: determinísticas)'
    )
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()