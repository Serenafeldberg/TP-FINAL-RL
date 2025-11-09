"""
Script de verificación rápida para testear la implementación de PPO.

Ejecuta un mini-entrenamiento de 1000 steps para verificar que todo funciona.
"""
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np

# Importar ale_py ANTES de gymnasium para registrar el namespace ALE
import ale_py
import gymnasium

print("=" * 60)
print("TEST DE VERIFICACIÓN PPO")
print("=" * 60)

# Test 1: Imports
print("\n[1/6] Verificando imports...")
try:
    from config import Config
    from envs.wrappers import make_env
    from ppoAgent.actorCritic import ActorCritic
    from ppoAgent.ppo import PPO
    from ppoAgent.memory import RolloutBuffer
    from ppoAgent.gae import compute_gae
    print("  ✓ Todos los imports OK")
except Exception as e:
    print(f"  ✗ Error en imports: {e}")
    sys.exit(1)

# Test 2: Environment
print("\n[2/6] Creando environment...")
try:
    env = make_env(
        env_id="ALE/Skiing-v5",
        seed=42,
        frame_stack=4,
        action_repeat=4,
        gray=True,
        resize_shape=(84, 84),
        normalize_obs=True,
        clip_rewards=True
    )
    obs, _ = env.reset(seed=42)
    print(f"  ✓ Environment creado")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
except Exception as e:
    print(f"  ✗ Error creando environment: {e}")
    sys.exit(1)

# Test 3: Actor-Critic
print("\n[3/6] Creando redes...")
try:
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    
    actor_critic = ActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        action_type="discrete",
        cnn_channels=(32, 64, 64),
        cnn_kernels=(8, 4, 3),
        cnn_strides=(4, 2, 1),
        hidden_size=512
    )
    
    total_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"  ✓ Actor-Critic creado")
    print(f"  Parámetros: {total_params:,}")
except Exception as e:
    print(f"  ✗ Error creando redes: {e}")
    sys.exit(1)

# Test 4: PPO Agent
print("\n[4/6] Inicializando PPO...")
try:
    agent = PPO(
        actor_critic=actor_critic,
        learning_rate=2.5e-4,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu"
    )
    print("  ✓ PPO inicializado")
except Exception as e:
    print(f"  ✗ Error inicializando PPO: {e}")
    sys.exit(1)

# Test 5: Rollout
print("\n[5/6] Probando rollout...")
try:
    buffer = RolloutBuffer(
        buffer_size=128,
        obs_shape=obs_shape,
        action_dim=1,
        device="cpu"
    )
    
    obs, _ = env.reset()
    
    for step in range(10):
        action, log_prob, value = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        buffer.add(obs, action, log_prob, reward, done, value)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
    
    print("  ✓ Rollout OK")
    print(f"  Steps recolectados: {buffer.pos}")
except Exception as e:
    print(f"  ✗ Error en rollout: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: GAE y Update
print("\n[6/6] Probando GAE y update...")
try:
    # Llenar el buffer
    obs, _ = env.reset()
    for step in range(buffer.pos, buffer.buffer_size):
        action, log_prob, value = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        buffer.add(obs, action, log_prob, reward, done, value)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
    
    # Compute advantages
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        last_value = agent.actor_critic.get_value(obs_tensor).item()
    
    buffer.compute_returns_and_advantage(
        last_value=last_value,
        last_done=done,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    print("  ✓ GAE calculado")
    
    # Update
    metrics = agent.update(
        rollout_buffer=buffer,
        batch_size=64,
        n_epochs=2,
        total_timesteps=1000,
        current_timestep=128
    )
    
    print("  ✓ Update OK")
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {-metrics['entropy_loss']:.4f}")
    
except Exception as e:
    print(f"  ✗ Error en GAE/update: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "=" * 60)
print("✓✓✓ TODOS LOS TESTS PASARON ✓✓✓")
print("=" * 60)
print("\nImplementación correcta! Listo para entrenar.")
print("Ejecuta: python mainTrainSkiing.py")
print("=" * 60)

env.close()