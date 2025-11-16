"""Script principal de entrenamiento PPO para Flappy Bird.

Soporta:
- --dry-run: Imprime configuración y sale
- --check-env: Crea el entorno y ejecuta 10 pasos random
- Sin flags: Ejecuta el entrenamiento completo de PPO
"""

import argparse
import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

import gymnasium as gym

from src.config import Config
from src.ppoAgent.utils import set_seed, print_system_info
from src.envs.wrappers import make_env
from src.ppoAgent.actorCritic import ActorCritic
from src.ppoAgent.ppo import PPO
from src.ppoAgent.memory import RolloutBuffer


def parse_args():
    p = argparse.ArgumentParser(description="PPO-Clip — Flappy Bird Training")
    p.add_argument("--dry-run", action="store_true",
                   help="Imprime configuración y sale (sin crear entorno).")
    p.add_argument("--check-env", action="store_true",
                   help="Crea el entorno y ejecuta 10 pasos random.")
    p.add_argument("--env-id", type=str, default=None,
                   help="Override del id de entorno (por defecto usa Config.ENV_NAME).")
    p.add_argument("--load-model", type=str, default=None,
                   help="Path a un checkpoint .pth para continuar entrenamiento.")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="Override de TOTAL_TIMESTEPS.")
    return p.parse_args()


def build_env(env_id: str, args=None):
    """Crea el entorno con la cadena de wrappers definida en Config.get_env_args()."""
    env_kwargs = Config.get_env_args()
    
    return make_env(
        env_id=env_id,
        seed=Config.SEED,
        **env_kwargs,
    )


def dry_run():
    """Imprime la configuración y la info de sistema."""
    Config.print_config()
    print_system_info()


def check_env(env):
    """Hace reset, imprime shape/dtype y ejecuta 10 pasos aleatorios."""
    print("\n=== CHECK ENV ===")
    res = env.reset()
    obs, info = (res if isinstance(res, tuple) else (res, {}))

    if isinstance(obs, np.ndarray):
        print(f"obs.shape: {obs.shape} | obs.dtype: {obs.dtype}")
        print(f"obs sample: {obs}")
    else:
        print(f"obs type: {type(obs)}")

    total_r = 0.0
    for t in range(10):
        action = env.action_space.sample()
        step_out = env.step(action)
        # Soportar firmas Gym vs Gymnasium
        if len(step_out) == 4:
            obs, rew, done, info = step_out
        else:
            obs, rew, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        total_r += float(rew)
        print(f"Step {t+1}: action={action}, reward={rew:.3f}, done={done}")
        if done:
            obs, _ = env.reset()
    print(f"10 steps OK | total_reward_accum={total_r:.3f}\n")


# Logger

class Logger:
    """Logger para tracking de métricas."""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de log
        self.rewards_file = self.log_dir / "rewards.csv"
        self.losses_file = self.log_dir / "losses.csv"
        
        # Inicializar archivos
        with open(self.rewards_file, 'w') as f:
            f.write("timestep,episode,reward,length\n")
        
        with open(self.losses_file, 'w') as f:
            f.write("timestep,policy_loss,value_loss,entropy_loss,clip_fraction,approx_kl,learning_rate\n")
    
    def log_episode(self, timestep, episode, reward, length):
        """Log info de episodio."""
        with open(self.rewards_file, 'a') as f:
            f.write(f"{timestep},{episode},{reward},{length}\n")
    
    def log_update(self, timestep, metrics):
        """Log métricas de update."""
        with open(self.losses_file, 'a') as f:
            f.write(
                f"{timestep},"
                f"{metrics['policy_loss']},"
                f"{metrics['value_loss']},"
                f"{metrics['entropy_loss']},"
                f"{metrics['clip_fraction']},"
                f"{metrics['approx_kl']},"
                f"{metrics['learning_rate']}\n"
            )

# Entrenamiento PPO

def train(env, args):
    """Función principal de entrenamiento PPO."""
    
    print("\n" + "=" * 60)
    print("INICIANDO ENTRENAMIENTO PPO - FLAPPY BIRD")
    print("=" * 60)
    
    # Obtener info del environment
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    action_type = "discrete" if hasattr(env.action_space, 'n') else "continuous"
    
    print(f"\n[1/5] Environment Info:")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action dim: {action_dim}")
    print(f"  Action type: {action_type}")
    
    # Crear Actor-Critic
    print(f"\n[2/5] Creando redes Actor-Critic (MLP)...")
    actor_critic = ActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        action_type=action_type,
        hidden_size=Config.HIDDEN_SIZE
    )
    
    total_params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
    print(f"  Total parámetros: {total_params:,}")
    
    # Crear agente PPO
    print(f"\n[3/5] Inicializando PPO...")
    agent = PPO(
        actor_critic=actor_critic,
        learning_rate=Config.LEARNING_RATE,
        clip_epsilon=Config.CLIP_EPSILON,
        value_loss_coef=Config.VALUE_LOSS_COEF,
        entropy_coef=Config.ENTROPY_COEF,
        max_grad_norm=Config.MAX_GRAD_NORM,
        lr_decay=Config.LR_DECAY,
        device=Config.DEVICE
    )
    
    # Cargar modelo si se especifica
    if args.load_model:
        print(f"  Cargando modelo desde: {args.load_model}")
        agent.load(args.load_model)
    
    # Crear buffer
    print(f"\n[4/5] Creando rollout buffer...")
    buffer_size = Config.N_STEPS * Config.N_ENVS
    if Config.BATCH_SIZE > buffer_size:
        print(f"WARNING: BATCH_SIZE ({Config.BATCH_SIZE}) > buffer_size ({buffer_size})")
        print(f"Adjusting BATCH_SIZE to {buffer_size}")
        Config.BATCH_SIZE = buffer_size
    
    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        obs_shape=obs_shape,
        action_dim=1 if action_type == "discrete" else action_dim,
        device=Config.DEVICE
    )
    
    # Logger
    logger = Logger(Config.LOG_DIR)
    
    # Total timesteps
    total_timesteps = args.total_timesteps if args.total_timesteps else Config.TOTAL_TIMESTEPS
    
    # Variables de tracking
    print(f"\n[5/5] Iniciando entrenamiento...")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Steps per rollout: {Config.N_STEPS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Epochs per update: {Config.N_EPOCHS}")
    print("=" * 60 + "\n")
    
    obs, _ = env.reset(seed=Config.SEED)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    
    start_time = time.time()
    
    # Loop principal de entrenamiento
    for timestep in range(1, total_timesteps + 1):
        
        # === ROLLOUT PHASE ===
        # Recolectar N_STEPS de experiencia, osea trayectorias usando la politica actual
        for step in range(Config.N_STEPS):
            #seleccionar acción (usando la politica actual)
            action, log_prob, value = agent.get_action(obs)
            
            #ejecuto la accion en el environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Almacenar en buffer la transición (s, a, r, s', done)
            buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob, #guardo log probabilidad de la accion
                reward=reward,
                done=done,
                value=value #valor estimado del estado
            )
            
            # Actualizar estado
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            # Si el episodio terminó
            if done:
                # Log episodio
                logger.log_episode(timestep, episode_count, episode_reward, episode_length)
                
                if episode_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = timestep / elapsed if elapsed > 0 else 0
                    print(
                        f"Timestep: {timestep:7,} | "
                        f"Episode: {episode_count:4d} | "
                        f"Reward: {episode_reward:8.2f} | "
                        f"Length: {episode_length:4d} | "
                        f"FPS: {fps:.0f}"
                    )
                
                # Reset
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
                episode_count += 1
        
        # === COMPUTE ADVANTAGES ===
        # Necesitamos el valor del último estado para bootstrap
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=Config.DEVICE).unsqueeze(0)
            last_value = agent.actor_critic.get_value(obs_tensor).item()
        
        # Compute returns y advantages usando GAE (combina TD y monte carlo)
        buffer.compute_returns_and_advantage(
            last_value=last_value,
            last_done=done,
            gamma=Config.GAMMA,
            gae_lambda=Config.GAE_LAMBDA
        )
        
        # === UPDATE PHASE ===
        # Actualizar policy por N_EPOCHS usando los datos recolectados
        metrics = agent.update(
            rollout_buffer=buffer,
            batch_size=Config.BATCH_SIZE,
            n_epochs=Config.N_EPOCHS,
            total_timesteps=total_timesteps,
            current_timestep=timestep
        )
        
        # Log metrics
        if timestep % Config.LOG_FREQ == 0:
            logger.log_update(timestep, metrics)
            entropy_val = -metrics['entropy_loss']
            print(f"\n[Update @ {timestep:,}]")
            print(f"  Policy Loss:   {metrics['policy_loss']:8.4f}")
            print(f"  Value Loss:    {metrics['value_loss']:8.4f}")
            print(f"  Entropy:       {entropy_val:8.4f}", end="")
            if entropy_val < 0.01:
                print("WARNING: Entropy muy bajo - política casi determinista!")
            else:
                print()
            print(f"  Clip Fraction: {metrics['clip_fraction']:8.3f}")
            print(f"  Approx KL:     {metrics['approx_kl']:8.4f}")
            print(f"  Learning Rate: {metrics['learning_rate']:.2e}\n")
        
        # Reset buffer para próximo rollout
        buffer.reset()
        
        # === SAVE MODEL ===
        if timestep % Config.SAVE_FREQ == 0:
            save_path = os.path.join(Config.MODEL_DIR, f"ppo_flappy_{timestep}.pth")
            agent.save(save_path)
    
    # Guardar modelo final
    final_path = os.path.join(Config.MODEL_DIR, "ppo_flappy_final.pth")
    agent.save(final_path)
    
    # Stats finales
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Tiempo total: {total_time/60:.1f} minutos")
    print(f"FPS promedio: {total_timesteps/total_time:.0f}")
    print(f"Episodios totales: {episode_count}")
    print("=" * 60)



if __name__ == "__main__":
    args = parse_args()

    # Dry-run: solo imprimir config
    if args.dry_run and not args.check_env:
        dry_run()
        sys.exit(0)

    # Seed global
    set_seed(Config.SEED)

    # Elegir env_id
    env_id = args.env_id or Config.ENV_NAME
    print(f"Target env_id: {env_id}")

    # Crear entorno
    try:
        env = build_env(env_id, args)
    except Exception as e:
        msg = str(e)
        print(f"[ERROR] No se pudo crear '{env_id}': {msg}")
        print(f"[INFO] Asegúrate de tener instalado el entorno Flappy Bird con LIDAR.")
        raise

    # Check-env: solo smoke test
    if args.check_env:
        check_env(env)
        env.close()
        sys.exit(0)

    # Entrenamiento completo
    try:
        train(env, args)
    finally:
        env.close()

