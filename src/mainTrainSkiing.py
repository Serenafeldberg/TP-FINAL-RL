import argparse
import sys
import numpy as np

import gymnasium as gym

from src.config import Config
from src.ppoAgent.utils import set_seed, print_system_info
from src.envs.wrappers import make_env

import ale_py


def parse_args():
    p = argparse.ArgumentParser(description="PPO-Clip — Fases 1 y 2 (dry-run / check-env)")
    p.add_argument("--dry-run", action="store_true",
                   help="Imprime configuración y sale (sin crear entorno).")
    p.add_argument("--check-env", action="store_true",
                   help="Crea el entorno y ejecuta 10 pasos random.")
    p.add_argument("--env-id", type=str, default=None,
                   help="Override del id de entorno (por defecto usa Config.ENV_NAME).")
    p.add_argument("--render", action="store_true",
                   help="Fuerza render_mode='rgb_array' si el entorno lo soporta.")
    return p.parse_args()


def build_env(env_id: str):
    """Crea el entorno con la cadena de wrappers definida en Config.get_env_args()."""
    env_kwargs = Config.get_env_args()
    # Permitir activar render rápido desde CLI sin tocar config
    if Config.RENDER_MODE is not None:
        env_kwargs["render_mode"] = Config.RENDER_MODE
    # --render pisa la config si se solicita
    if "render_mode" not in env_kwargs and args.render:
        env_kwargs["render_mode"] = "rgb_array"

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
    else:
        print(f"obs type: {type(obs)}")

    total_r = 0.0
    for t in range(10):
        action = env.action_space.sample()
        step_out = env.step(action)
        # Soportar firmas Gym vs Gymnasium
        print(step_out)
        if len(step_out) == 4:
            obs, rew, done, info = step_out
        else:
            obs, rew, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        total_r += float(rew)
        if done:
            env.reset()
    print(f"10 steps OK | total_reward_accum={total_r:.3f}\n")


if __name__ == "__main__":
    args = parse_args()

    if args.dry_run and not args.check_env:
        dry_run()
        sys.exit(0)

    # Seed global
    set_seed(Config.SEED)

    # Elegir env_id
    env_id = args.env_id or Config.ENV_NAME
    print(f"Target env_id: {env_id}")

    # Crear entorno (con fallback si falta ALE/Box2D)
    try:
        env = build_env(env_id)
    except Exception as e:
        msg = str(e)
        print(f"[WARN] No se pudo crear '{env_id}': {msg}")

        # Fallbacks prácticos para testear wrappers sin instalar ROMs de Atari
        fallback = None
        if "ALE" in env_id or "Namespace ALE not found" in msg:
            fallback = "CarRacing-v3"  # visual sin ROMs (requiere box2d)
        if "Box2D" in msg or "DependencyNotInstalled" in msg:
            # Si falla Box2D, último recurso: pixeles sobre CartPole
            fallback = "CartPole-v1"
            print("[INFO] Probando CartPole con PixelObservation (render_mode='rgb_array').")

        if fallback is None:
            raise  # no sabemos cómo recuperar

        # Ajuste de kwargs para fallback CartPole con pixeles
        if fallback == "CartPole-v1":
            # Forzar render a pixeles para que PreprocessObs/FrameStack tengan sentido
            Config.RENDER_MODE = "rgb_array"

        print(f"[INFO] Fallback -> {fallback}")
        env = build_env(fallback)

    # Si sólo querías chequear el entorno, corré smoke test
    if args.check_env:
        check_env(env)
        sys.exit(0)

    # Si no pasaste flags, por ahora hacemos dry-run por seguridad
    dry_run()
