"""
Script para evaluar modelos PPO entrenados.

Uso:
    python evaluate.py --model path/to/model.pth --episodes 10 --render
    python evaluate.py --model path/to/model.pth --record-video --episodes 1
"""
import sys
from pathlib import Path
import argparse

src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from config import Config
from envs.wrappers import make_env
from ppoAgent.actorCritic import ActorCritic
from ppoAgent.ppo import PPO


def record_video(
    model_path: str,
    n_episodes: int = 1,
    video_name: str = "trainedAgent",
    deterministic: bool = True
):
    """
    Grabar video del agente jugando.
    
    Args:
        model_path: path al checkpoint .pth
        n_episodes: número de episodios a grabar
        video_name: nombre del archivo de video (sin extensión)
        deterministic: si usar acciones determinísticas
    """
    print("=" * 60)
    print("GRABANDO VIDEO DEL AGENTE")
    print("=" * 60)
    print(f"Modelo: {model_path}")
    print(f"Episodios: {n_episodes}")
    print(f"Video se guardará en: {Config.VIDEO_DIR}")
    print("=" * 60)
    
    # Crear entorno con render_mode para video
    print("\n[1/3] Creando entorno...")
    
    try:
        import ale_py
    except ImportError:
        pass
    
    # Crear entorno base con render
    env = gym.make(Config.ENV_NAME, render_mode='rgb_array')
    
    # Aplicar wrappers básicos (sin preprocesamiento visual intenso)
    from envs.wrappers import StepAPICompat, ActionRepeat, RewardClip, TimeLimit
    
    env = StepAPICompat(env)
    
    if Config.ACTION_REPEAT > 1:
        env = ActionRepeat(env, repeat=Config.ACTION_REPEAT)
    
    if Config.MAX_EPISODE_STEPS:
        env = TimeLimit(env, Config.MAX_EPISODE_STEPS)
    
    if Config.CLIP_REWARDS:
        env = RewardClip(env)
    
    # Wrapper de RecordVideo
    env = RecordVideo(
        env,
        video_folder=Config.VIDEO_DIR,
        name_prefix=video_name,
        episode_trigger=lambda x: True,  # Grabar todos los episodios
        video_length=0,  # Grabar episodios completos
    )
    
    # Ahora aplicar wrappers de preprocesamiento (para que el agente entienda)
    from envs.wrappers import PreprocessObs, FrameStack
    
    if Config.GRAYSCALE:
        env = PreprocessObs(
            env,
            gray=Config.GRAYSCALE,
            resize_shape=(Config.FRAME_SIZE, Config.FRAME_SIZE),
            normalize=Config.NORMALIZE_OBS
        )
    
    if Config.FRAME_STACK > 1:
        env = FrameStack(env, k=Config.FRAME_STACK)
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    action_type = "discrete" if hasattr(env.action_space, 'n') else "continuous"
    
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action dim: {action_dim}")
    
    # Cargar modelo
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
    agent.actor_critic.eval()
    
    # Grabar episodios
    print("\n[3/3] Grabando video...")
    print("-" * 60)
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        seed = Config.SEED + episode if Config.SEED else None
        obs, _ = env.reset(seed=seed)
        
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # Usar política para decidir acción
            action, _, _ = agent.get_action(obs, deterministic=deterministic)
            
            # Ejecutar acción
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        
        print(
            f"Episode {episode + 1:2d}/{n_episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Length: {episode_length:4d}"
        )
    
    env.close()
    
    print("-" * 60)
    print(f"\n✓ Video guardado en: {Config.VIDEO_DIR}")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    # Listar videos generados
    import os
    videos = [f for f in os.listdir(Config.VIDEO_DIR) if f.endswith('.mp4')]
    if videos:
        print(f"  Videos generados:")
        for v in sorted(videos)[-n_episodes:]:  # Mostrar últimos n
            video_path = os.path.join(Config.VIDEO_DIR, v)
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"    - {v} ({size_mb:.1f} MB)")
    
    print("=" * 60)


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
    parser.add_argument(
        '--record-video',
        action='store_true',
        help='Grabar video del agente jugando'
    )
    parser.add_argument(
        '--video-name',
        type=str,
        default='trainedAgent',
        help='Nombre del archivo de video (default: trainedAgent)'
    )
    
    args = parser.parse_args()
    
    if args.record_video:
        record_video(
            model_path=args.model,
            n_episodes=args.episodes,
            video_name=args.video_name,
            deterministic=not args.stochastic
        )
    else:
        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            render=args.render,
            deterministic=not args.stochastic
        )


if __name__ == "__main__":
    main()