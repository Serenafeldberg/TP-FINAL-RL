"""Script principal de entrenamiento PPO para Flappy Bird con búsqueda de hiperparámetros.

Soporta:
- --dry-run: Imprime configuración y sale
- --check-env: Crea el entorno y ejecuta 10 pasos random
- --config-id: ID de configuración (1-5) o 'all' para todas
- --test-mode: Modo de prueba con muy pocos timesteps (default: False)
- Sin flags: Ejecuta todas las configuraciones
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import torch
from pathlib import Path

import gymnasium as gym

import flappy_bird_gymnasium
from config import Config
from ppoAgent.utils import set_seed, print_system_info
#from envs.wrappers import make_env
from ppoAgent.actorCritic import ActorCritic
from ppoAgent.ppo import PPO
from ppoAgent.memory import RolloutBuffer


# ============================================================================
# CONFIGURACIONES DE HIPERPARÁMETROS
# ============================================================================
HYPERPARAM_CONFIGS = [
    {
        'id': 1,
        'name': 'Baseline',
        'learning_rate': 1e-4,
        'entropy_coef': 0.05,
        'n_epochs': 4,
        'hidden_size': 256,
        'batch_size': 64,
    },
    {
        'id': 2,
        'name': 'Mayor_Entropia',
        'learning_rate': 1e-4,
        'entropy_coef': 0.1,
        'n_epochs': 4,
        'hidden_size': 256,
        'batch_size': 64,
    },
    {
        'id': 3,
        'name': 'LR_Alto',
        'learning_rate': 3e-4,
        'entropy_coef': 0.05,
        'n_epochs': 4,
        'hidden_size': 256,
        'batch_size': 64,
    },
    {
        'id': 4,
        'name': 'Red_Grande',
        'learning_rate': 1e-4,
        'entropy_coef': 0.05,
        'n_epochs': 4,
        'hidden_size': 512,
        'batch_size': 64,
    },
    {
        'id': 5,
        'name': 'Mas_Epocas',
        'learning_rate': 1e-4,
        'entropy_coef': 0.05,
        'n_epochs': 8,
        'hidden_size': 256,
        'batch_size': 64,
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="PPO-Clip — Flappy Bird Training con Búsqueda de Hiperparámetros")
    p.add_argument("--dry-run", action="store_true",
                   help="Imprime configuración y sale (sin crear entorno).")
    p.add_argument("--check-env", action="store_true",
                   help="Crea el entorno y ejecuta 10 pasos random.")
    p.add_argument("--config-id", type=str, default="all",
                   help="ID de configuración (1-5) o 'all' para todas (default: all)")
    p.add_argument("--test-mode", action="store_true",
                   help="Modo de prueba con muy pocos timesteps (5 updates)")
    p.add_argument("--env-id", type=str, default=None,
                   help="Override del id de entorno (por defecto usa Config.ENV_NAME).")
    p.add_argument("--load-model", type=str, default=None,
                   help="Path a un checkpoint .pth para continuar entrenamiento.")
    return p.parse_args()


#def build_env(env_id: str, args=None):
#    """Crea el entorno con la cadena de wrappers definida en Config.get_env_args()."""
#    env_kwargs = Config.get_env_args()
#    
#    return make_env(
#        env_id=env_id,
#        seed=Config.SEED,
#        **env_kwargs,
#    )


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

def train(env, args, config_dict=None):
    """Función principal de entrenamiento PPO."""
    
    config_name = config_dict['name'] if config_dict else "Default"
    print("\n" + "=" * 60)
    print(f"INICIANDO ENTRENAMIENTO PPO - FLAPPY BIRD - {config_name}")
    print("=" * 60)
    
    # Aplicar configuración de hiperparámetros si se proporciona
    if config_dict:
        Config.LEARNING_RATE = config_dict['learning_rate']
        Config.ENTROPY_COEF = config_dict['entropy_coef']
        Config.N_EPOCHS = config_dict['n_epochs']
        Config.HIDDEN_SIZE = config_dict['hidden_size']
        Config.BATCH_SIZE = config_dict['batch_size']
        
        print(f"\nConfiguración aplicada:")
        print(f"  Learning Rate: {Config.LEARNING_RATE}")
        print(f"  Entropy Coef: {Config.ENTROPY_COEF}")
        print(f"  N Epochs: {Config.N_EPOCHS}")
        print(f"  Hidden Size: {Config.HIDDEN_SIZE}")
        print(f"  Batch Size: {Config.BATCH_SIZE}")
    
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
    
    # Logger - usar directorio específico de la configuración
    if config_dict:
        # Guardar en savedModels/search/config_N/
        config_model_dir = os.path.join(Config.MODEL_DIR, "search", f"config_{config_dict['id']}_{config_dict['name']}")
        config_log_dir = os.path.join(config_model_dir, "logs")
        os.makedirs(config_model_dir, exist_ok=True)
        os.makedirs(config_log_dir, exist_ok=True)
        Config.MODEL_DIR = config_model_dir
        Config.LOG_DIR = config_log_dir
        
        # Guardar configuración en JSON
        config_file = os.path.join(config_model_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"  → Modelos se guardarán en: {config_model_dir}")
    
    logger = Logger(Config.LOG_DIR)
    
    # Total timesteps - modo test con muy pocos updates
    # Nota: el loop principal itera sobre "updates", cada uno recolecta N_STEPS de experiencia
    if args.test_mode:
        # Solo 5 updates para verificar que funciona
        Config.N_STEPS = 5
        total_timesteps = 5  # 5 updates = 5 * N_STEPS timesteps reales del entorno
        print(f"\n⚠️  MODO TEST: Solo {total_timesteps} updates ({total_timesteps * Config.N_STEPS:,} timesteps del entorno)")
    else:
        # Calcular número de updates basado en TOTAL_TIMESTEPS
        # TOTAL_TIMESTEPS es el número total de timesteps del entorno que queremos
        total_timesteps = Config.TOTAL_TIMESTEPS // Config.N_STEPS
        print(f"\n📊 Entrenamiento completo: {total_timesteps:,} updates ({Config.TOTAL_TIMESTEPS:,} timesteps del entorno)")
    
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
            #obs = obs / 400.0   # si tus distancias están en px (0–400)
            obs = (obs - obs.mean()) / (obs.std() + 1e-8)
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
        if not args.test_mode and timestep % Config.SAVE_FREQ == 0:
            save_path = os.path.join(Config.MODEL_DIR, f"ppo_flappy_{timestep}.pth")
            agent.save(save_path)
    
    # Guardar modelo final
    model_name = f"ppo_flappy_{config_name}_final.pth" if config_dict else "ppo_flappy_final.pth"
    final_path = os.path.join(Config.MODEL_DIR, model_name)
    agent.save(final_path)
    print(f"\n✓ Modelo final guardado en: {final_path}")
    
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
        print("\n" + "=" * 60)
        print("CONFIGURACIONES DISPONIBLES:")
        print("=" * 60)
        for config in HYPERPARAM_CONFIGS:
            print(f"\nConfig {config['id']}: {config['name']}")
            print(f"  LR: {config['learning_rate']}, Entropy: {config['entropy_coef']}, "
                  f"Epochs: {config['n_epochs']}, Hidden: {config['hidden_size']}")
        sys.exit(0)

    # Seed global
    set_seed(Config.SEED)

    # Elegir env_id
    env_id = args.env_id or Config.ENV_NAME
    print(f"Target env_id: {env_id}")

    # Determinar qué configuraciones ejecutar
    if args.config_id.lower() == "all":
        configs_to_run = HYPERPARAM_CONFIGS
    else:
        try:
            config_id = int(args.config_id)
            configs_to_run = [c for c in HYPERPARAM_CONFIGS if c['id'] == config_id]
            if not configs_to_run:
                print(f"❌ Error: Config ID {config_id} no encontrado. Usa 1-5 o 'all'")
                sys.exit(1)
        except ValueError:
            print(f"❌ Error: --config-id debe ser un número (1-5) o 'all'")
            sys.exit(1)

    print(f"\n{'='*80}")
    print(f"EJECUTANDO {len(configs_to_run)} CONFIGURACIÓN(ES)")
    if args.test_mode:
        print("⚠️  MODO TEST ACTIVADO (muy pocos timesteps)")
    print(f"{'='*80}\n")

    # Crear entorno una vez (se reutiliza)
    env = gym.make(Config.ENV_NAME, render_mode="human", use_lidar=True)

    # Check-env: solo smoke test
    if args.check_env:
        check_env(env)
        env.close()
        sys.exit(0)

    # Ejecutar cada configuración
    for config in configs_to_run:
        print(f"\n{'#'*80}")
        print(f"CONFIGURACIÓN {config['id']}/5: {config['name']}")
        print(f"{'#'*80}\n")
        
        try:
            train(env, args, config_dict=config)
        except Exception as e:
            print(f"\n❌ ERROR en configuración {config['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    env.close()
    
    print(f"\n{'='*80}")
    print("TODAS LAS CONFIGURACIONES COMPLETADAS")
    print(f"Resultados guardados en: {os.path.join(Config.MODEL_DIR, 'search')}")
    print(f"{'='*80}\n")

