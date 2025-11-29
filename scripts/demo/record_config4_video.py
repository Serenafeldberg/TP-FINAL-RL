"""
Script especializado para grabar video de Config 4 (mejor modelo).
Infiere automáticamente hidden_size=512 del checkpoint.
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium.wrappers import RecordVideo
from config import Config
from envs.wrappers import StepAPICompat, RewardClip, TimeLimit
from ppoAgent.actorCritic import ActorCritic
from ppoAgent.ppo import PPO


def record_video_config4(
    n_episodes: int = 3,
    video_name: str = "config4_best_model",
    deterministic: bool = True
):
    """
    Grabar video del mejor modelo (Config 4).
    """
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "savedModels/config_4_Red_Grande/ppo_flappy_Red_Grande_final.pth"
    config_path = project_root / "savedModels/config_4_Red_Grande/config.json"
    norm_path = project_root / "savedModels/config_4_Red_Grande/obs_norm_stats_Red_Grande.npz"
    
    print("=" * 70)
    print("GRABANDO VIDEO - CONFIG 4 (MEJOR MODELO)")
    print("=" * 70)
    print(f"Modelo: {model_path}")
    print(f"Episodios: {n_episodes}")
    print(f"Video se guardará en: {Config.VIDEO_DIR}")
    print("=" * 70)
    
    with open(str(config_path), 'r') as f:
        config_data = json.load(f)
    
    hidden_size = config_data.get('hidden_size', 512)
    print(f"\n[INFO] Hidden size del modelo: {hidden_size}")
    
    # crear entorno con render para video
    print("\n[1/4] Creando entorno...")
    
    env = gym.make(Config.ENV_NAME, render_mode='rgb_array', use_lidar=True)
    
    # aplicar wrappers basicos
    env = StepAPICompat(env)
    
    if Config.get_env_args().get('max_episode_steps'):
        env = TimeLimit(env, Config.get_env_args()['max_episode_steps'])
    
    if Config.get_env_args().get('clip_rewards'):
        env = RewardClip(env)
    
    # wrapper de RecordVideo
    env = RecordVideo(
        env,
        video_folder=Config.VIDEO_DIR,
        name_prefix=video_name,
        episode_trigger=lambda x: True,  
        video_length=0,  
    )
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    action_type = "discrete" if hasattr(env.action_space, 'n') else "continuous"
    
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hidden size: {hidden_size}")
    
    print("\n[2/4] Cargando modelo...")
    actor_critic = ActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        action_type=action_type,
        hidden_size=hidden_size  
    )
    
    agent = PPO(
        actor_critic=actor_critic,
        device=Config.DEVICE
    )
    
    agent.load(str(model_path))
    agent.actor_critic.eval()
    print("  ✓ Modelo cargado exitosamente")
    
    print("\n[3/4] Cargando normalización...")
    try:
        stats = np.load(str(norm_path))
        obs_norm_mean = stats['mean']
        obs_norm_std = np.sqrt(stats['var'])
        print(f"  Normalización cargada (shape: {obs_norm_mean.shape})")
    except Exception as e:
        print(f"  No se pudo cargar normalización: {e}")
        obs_norm_mean = None
        obs_norm_std = None
    
    print("\n[4/4] Grabando video...")
    print("-" * 70)
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []  # pipes atravesados
    
    for episode in range(n_episodes):
        seed = Config.SEED + episode if Config.SEED else None
        obs, info = env.reset(seed=seed)
        
        done = False
        episode_reward = 0.0
        episode_length = 0
        pipes_passed = 0
        
        while not done:
            # normalizar observacion
            if obs_norm_mean is not None and obs_norm_std is not None:
                obs_normalized = (np.asarray(obs, dtype=np.float32) - obs_norm_mean) / (obs_norm_std + 1e-8)
            else:
                obs_normalized = obs
            
            action, _, _ = agent.get_action(obs_normalized, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if 'score' in info:
                pipes_passed = info['score']
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(pipes_passed)
        
        print(
            f"Episode {episode + 1:2d}/{n_episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Pipes: {pipes_passed:3d} | "
            f"Length: {episode_length:4d}"
        )
    
    env.close()
    
    print("-" * 70)
    print("\nVIDEO GRABADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\nESTADISTICAS:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Pipes:  {np.mean(episode_scores):.1f} ± {np.std(episode_scores):.1f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    # listar videos generados
    import os
    print(f"\nVIDEOS GENERADOS en: {Config.VIDEO_DIR}")
    videos = [f for f in os.listdir(Config.VIDEO_DIR) if f.endswith('.mp4')]
    if videos:
        for v in sorted(videos)[-n_episodes:]:  # mostrar ultimos n
            video_path = os.path.join(Config.VIDEO_DIR, v)
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  {v} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    record_video_config4(
        n_episodes=3,
        video_name="config4_best_model",
        deterministic=True
    )

