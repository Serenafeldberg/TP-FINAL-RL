"""
Entrenamiento y evaluación de PPO en CartPole-v1 usando tu implementación actual.
Objetivo: verificar que el algoritmo aprende correctamente en un entorno sencillo.

Uso:
    PARA ENTRENAR:
    python src/test_cartpole.py --train

    PARA EVALUAR:
    python src/test_cartpole.py --eval --model savedModels/ppo_cartpole_final.pth --episodes 10
    (se puede cambiar el modelo por el que se quiera evaluar)

    PARA VERLO EN TIEMPO REAL: 
    python src/test_cartpole.py --eval --model savedModels/ppo_cartpole_final.pth --render --episodes 5
    
    PARA GRABAR VIDEO:
    python src/test_cartpole.py --eval --model savedModels/ppo_cartpole_final.pth --record-video --episodes 3
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

from ppoAgent.utils import set_seed
from ppoAgent.actorCritic import ActorCritic
from ppoAgent.ppo import PPO
from ppoAgent.memory import RolloutBuffer


# Script principal
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

    # Crear modelo y agente PPO
    # Usar arquitectura más pequeña para CartPole (más eficiente)
    model = ActorCritic(
        obs_shape, 
        action_dim, 
        action_type="discrete",
        hidden_size=64  
    ).to(device)
    agent = PPO(
        actor_critic=model,
        learning_rate=3e-4,       
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,        
        max_grad_norm=0.5,
        lr_decay=True,
        device=device,
    )

    # Buffer y parámetros de entrenamiento
    n_steps = 2048               
    batch_size = 64
    n_epochs = 10
    total_updates = 50           
    gamma = 0.99
    gae_lambda = 0.95
    
    # Umbral de éxito: CartPole resuelto si promedio > 475
    SUCCESS_THRESHOLD = 475

    buffer = RolloutBuffer(
        buffer_size=n_steps,
        obs_shape=obs_shape,
        action_dim=1,
        device=device
    )

    # Entrenamiento
    obs, _ = env.reset(seed=42)

    episode_rewards = []
    episode_lengths = []
    rewards_per_update = []

    current_return, current_len = 0, 0
    total_timesteps = 0

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
                action=action.item(),  # Convertir a escalar para acciones discretas
                log_prob=logp.item() if isinstance(logp, torch.Tensor) else logp,
                reward=torch.tensor([reward], dtype=torch.float32, device=device),
                done=torch.tensor([done], dtype=torch.float32, device=device),
                value=value.item() if isinstance(value, torch.Tensor) else value
            )

            current_return += reward
            current_len += 1

            obs = next_obs

            if done:
                episode_rewards.append(current_return)
                episode_lengths.append(current_len)
                current_return, current_len = 0, 0
                obs, _ = env.reset()

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

        total_timesteps += n_steps
        
        metrics = agent.update(
            rollout_buffer=buffer,
            batch_size=batch_size,
            n_epochs=n_epochs,
            total_timesteps=total_updates * n_steps,
            current_timestep=total_timesteps
        )

        buffer.reset()

        # Calcular promedio de últimos 10 episodios (o todos si hay menos)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        rewards_per_update.append(avg_reward)
        
        # Estadísticas adicionales
        recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
        std_reward = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0

        print(f"Update {update:03d}/{total_updates} | Timesteps: {total_timesteps:6d} | "
              f"AvgR: {avg_reward:6.1f} ± {std_reward:5.2f} | "
              f"PolLoss: {metrics['policy_loss']:.4f} | ValLoss: {metrics['value_loss']:.4f} | "
              f"Ent: {-metrics['entropy_loss']:.4f} | ClipFrac: {metrics['clip_fraction']:.3f}")

        # Guardar checkpoint cada 10 updates
        if update % 10 == 0:
            model_dir = Path(__file__).parent.parent / "savedModels"
            model_dir.mkdir(exist_ok=True)
            checkpoint_path = model_dir / f"ppo_cartpole_{total_timesteps}.pth"
            agent.save(str(checkpoint_path))
            print(f"  → Checkpoint guardado: {checkpoint_path}")

        # Verificar si está resuelto (promedio > 475)
        if avg_reward >= SUCCESS_THRESHOLD:
            print(f"\n{'='*60}")
            print(f"¡CartPole RESUELTO! 🎉")
            print(f"Mean Reward: {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"Umbral objetivo: {SUCCESS_THRESHOLD}+ ✅")
            print(f"Timesteps totales: {total_timesteps}")
            print(f"{'='*60}\n")
            # Guardar modelo final
            model_dir = Path(__file__).parent.parent / "savedModels"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "ppo_cartpole_final.pth"
            agent.save(str(model_path))
            print(f"Modelo final guardado en: {model_path}")
            break

    env.close()

    # Guardar modelo final si no se guardó antes
    if len(rewards_per_update) > 0:
        final_avg_reward = rewards_per_update[-1]
        final_std = np.std(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0.0
        if final_avg_reward < SUCCESS_THRESHOLD:
            model_dir = Path(__file__).parent.parent / "savedModels"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / "ppo_cartpole_final.pth"
            agent.save(str(model_path))
            print(f"\nModelo guardado en: {model_path}")
            print(f"Recompensa final: {final_avg_reward:.2f} ± {final_std:.2f} (objetivo: {SUCCESS_THRESHOLD}+)")
    else:
        # Si no hubo updates, guardar de todas formas
        model_dir = Path(__file__).parent.parent / "savedModels"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "ppo_cartpole_final.pth"
        agent.save(str(model_path))
        print(f"\nModelo guardado en: {model_path}")

    # Estadísticas finales
    if len(episode_rewards) > 0:
        print(f"\n{'='*60}")
        print("ESTADÍSTICAS FINALES:")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Total episodios: {len(episode_rewards)}")
        print(f"  Mean Reward (últimos 10): {np.mean(episode_rewards[-10:]):.2f} ± {np.std(episode_rewards[-10:]):.2f}")
        print(f"  Mean Reward (todos): {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Max Reward: {np.max(episode_rewards):.2f}")
        print(f"  Min Reward: {np.min(episode_rewards):.2f}")
        print(f"{'='*60}")

    # Graficar curva de recompensa
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_update, label="Recompensa promedio (últimos 10 episodios)", linewidth=2)
    plt.axhline(SUCCESS_THRESHOLD, color="r", linestyle="--", label=f"Umbral resuelto ({SUCCESS_THRESHOLD})", linewidth=2)
    plt.axhline(500, color="g", linestyle=":", label="Máximo posible (500)", alpha=0.7)
    plt.xlabel("Update", fontsize=12)
    plt.ylabel("Recompensa promedio", fontsize=12)
    plt.title("Curva de aprendizaje PPO - CartPole-v1", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar gráfico
    plot_dir = Path(__file__).parent / "plots" / "output"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "cartpole_training.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nGráfico guardado en: {plot_path}")
    plt.show()


def evaluate_cartpole(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    record_video: bool = False,
    video_name: str = "cartpole_agent",
    deterministic: bool = True,
    seed: int = 42
):
    """
    Evaluar un modelo PPO entrenado en CartPole-v1.
    
    Args:
        model_path: path al modelo .pth
        n_episodes: número de episodios a evaluar
        render: si renderizar en tiempo real
        record_video: si grabar video
        video_name: nombre del archivo de video
        deterministic: si usar acciones determinísticas
        seed: semilla para reproducibilidad
    """
    print("=" * 60)
    print("EVALUACIÓN DE MODELO PPO - CARTPOLE-V1")
    print("=" * 60)
    print(f"Modelo: {model_path}")
    print(f"Episodios: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print("=" * 60)
    
    set_seed(seed)
    device = torch.device("cpu")
    
    # Crear entorno
    print("\n[1/3] Creando entorno...")
    if record_video:
        video_dir = Path(__file__).parent / "logs" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.make("CartPole-v1", render_mode='rgb_array')
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            name_prefix=video_name,
            episode_trigger=lambda x: True,
            video_length=0,
        )
    elif render:
        env = gym.make("CartPole-v1", render_mode='human')
    else:
        env = gym.make("CartPole-v1")
    
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action dim: {action_dim}")
    
    # Cargar modelo
    print("\n[2/3] Cargando modelo...")
    model = ActorCritic(
        obs_shape, 
        action_dim, 
        action_type="discrete",
        hidden_size=64  # Misma arquitectura que en entrenamiento
    ).to(device)
    agent = PPO(actor_critic=model, device=device)
    agent.load(model_path)
    agent.actor_critic.eval()
    
    # Evaluar
    print("\n[3/3] Evaluando...")
    print("-" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode if seed else None)
        
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            action, _, _ = agent.get_action(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render and not record_video:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(
            f"Episode {episode + 1:2d}/{n_episodes} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Length: {episode_length:4d}"
        )
    
    env.close()
    
    # Estadísticas
    print("-" * 60)
    print("\nESTADÍSTICAS:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min Reward:  {np.min(episode_rewards):.2f}")
    print(f"  Max Reward:  {np.max(episode_rewards):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    if record_video:
        print(f"\n✓ Videos guardados en: {video_dir}")
        videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if videos:
            print(f"  Videos generados:")
            for v in sorted(videos)[-n_episodes:]:
                video_path = os.path.join(video_dir, v)
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"    - {v} ({size_mb:.1f} MB)")
    
    print("=" * 60)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }


def main():
    parser = argparse.ArgumentParser(description='Entrenar o evaluar PPO en CartPole-v1')
    parser.add_argument(
        '--train',
        action='store_true',
        help='Entrenar el modelo'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluar un modelo guardado'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path al modelo .pth (requerido para --eval)'
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
        help='Renderizar el entorno durante evaluación'
    )
    parser.add_argument(
        '--record-video',
        action='store_true',
        help='Grabar video del agente jugando'
    )
    parser.add_argument(
        '--video-name',
        type=str,
        default='cartpole_agent',
        help='Nombre del archivo de video (default: cartpole_agent)'
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Usar acciones estocásticas (default: determinísticas)'
    )
    
    args = parser.parse_args()
    
    if args.eval:
        if args.model is None:
            # Intentar usar modelo por defecto
            default_model = Path(__file__).parent.parent / "savedModels" / "ppo_cartpole_final.pth"
            if default_model.exists():
                args.model = str(default_model)
                print(f"Usando modelo por defecto: {args.model}")
            else:
                print("Error: --model es requerido para --eval")
                parser.print_help()
                return
        
        evaluate_cartpole(
            model_path=args.model,
            n_episodes=args.episodes,
            render=args.render,
            record_video=args.record_video,
            video_name=args.video_name,
            deterministic=not args.stochastic
        )
    elif args.train:
        train_cartpole()
    else:
        # Por defecto, entrenar
        print("No se especificó --train ni --eval, entrenando por defecto...")
        train_cartpole()


if __name__ == "__main__":
    main()