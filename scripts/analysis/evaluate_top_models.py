#!/usr/bin/env python3
"""
evaluar los agentes PPO de mejor rendimiento entrenados en Flappy Bird.

Workflow:
    1. Reutilizar el ranking calculado por ``analyze_flappy_results`` para seleccionar las TOP-K configuraciones.
    2. Para cada configuracion, ubicar el checkpoint final y las estadisticas de normalizacion (.npz).
    3. Ejecutar rollouts deterministicos (50 episodios por defecto) en Flappy Bird, reutilizando la misma preprocesamiento de observaciones que durante el entrenamiento (stats-based cuando esta disponible, de lo contrario se vuelve a obs/400.0 scaling como se documenta en el proyecto).
    4. Exportar metricas agregadas (mean/std/min/max reward) a ``evaluation_top_models.csv``.

ejecucion:
    python evaluate_top_models.py --top-k 3 --episodes 50
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import gymnasium as gym
import flappy_bird_gymnasium as _  # noqa: F401  # Ensures env is registered

import analyze_flappy_results as analyzer

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config  # type: ignore  # noqa: E402
from ppoAgent.actorCritic import ActorCritic  # type: ignore  # noqa: E402
from ppoAgent.ppo import PPO  # type: ignore  # noqa: E402


DEFAULT_SAVED_MODELS_DIR = BASE_DIR / "savedModels"
DEFAULT_OUTPUT_PATH = BASE_DIR / "evaluation_top_models.csv"
ROLLING_WINDOW = 100
DEFAULT_TOP_K = 3
DEFAULT_EPISODES = 50
NORMALIZATION_SCALE = 400.0  # fallback scaling when stats are unavailable


@dataclass
class EvalResult:
    rank: int
    config_id: int
    config_name: str
    model_path: str
    stats_path: str
    hidden_size: int
    learning_rate: float
    entropy_coef: float
    n_epochs: int
    batch_size: int
    episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    mean_score: float
    normalization: str

    def to_record(self) -> Dict[str, object]:
        return asdict(self)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the top PPO agents trained on Flappy Bird."
    )
    parser.add_argument(
        "--saved-models-dir",
        type=Path,
        default=DEFAULT_SAVED_MODELS_DIR,
        help="Root directory containing config_* folders (default: %(default)s)",
    )
    parser.add_argument(
        "--max-config-id",
        type=int,
        default=18,
        help="Highest configuration id to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of best configurations to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Episodes per configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path (default: %(default)s)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)",
    )
    parser.add_argument(
        "--allow-stochastic",
        action="store_true",
        help="If set, allows stochastic actions (overrides deterministic flag).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=Config.DEVICE,
        help="Device used to run inference (default: %(default)s)",
    )
    return parser.parse_args(argv)


def locate_model_file(config_dir: Path) -> Optional[Path]:
    """Return the final checkpoint inside a configuration directory."""
    candidates = sorted(config_dir.glob("ppo_flappy*_final.pth"))
    if not candidates:
        candidates = sorted(config_dir.glob("*.pth"))
    return candidates[-1] if candidates else None


def locate_stats_file(config_dir: Path) -> Optional[Path]:
    candidates = sorted(config_dir.glob("obs_norm_stats*.npz"))
    return candidates[-1] if candidates else None


def load_obs_stats(stats_path: Optional[Path]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load mean/std arrays from the saved RunningNorm statistics."""
    if stats_path is None or not stats_path.exists():
        return None
    try:
        data = np.load(stats_path)
        mean = data.get("mean")
        if mean is None:
            return None
        if "std" in data:
            std = data["std"]
        elif "var" in data:
            std = np.sqrt(data["var"])
        else:
            return None
        return mean.astype(np.float32), std.astype(np.float32)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[WARN] Could not load stats from {stats_path}: {exc}")
        return None


def normalize_obs(obs: np.ndarray, stats: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if stats is not None:
        mean, std = stats
        try:
            return ((arr - mean) / (std + 1e-8)).astype(np.float32)
        except ValueError:
            mean_b = np.broadcast_to(mean, arr.shape)
            std_b = np.broadcast_to(std, arr.shape)
            return ((arr - mean_b) / (std_b + 1e-8)).astype(np.float32)
    return arr / NORMALIZATION_SCALE


def create_env() -> gym.Env:
    """instanciar el entorno Flappy Bird sin renderizado."""
    env = gym.make(Config.ENV_NAME, render_mode=None, use_lidar=True)
    return env


def build_agent(
    env: gym.Env,
    hidden_size: int,
    device: str,
) -> PPO:
    """instanciar PPO con la red correcta para una configuracion dada."""
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    action_type = "discrete" if hasattr(env.action_space, "n") else "continuous"

    actor_critic = ActorCritic(
        obs_shape=obs_shape,
        action_dim=action_dim,
        action_type=action_type,
        hidden_size=hidden_size,
    )
    agent = PPO(actor_critic=actor_critic, device=device)
    agent.actor_critic.eval()
    return agent


def run_rollouts(
    agent: PPO,
    env: gym.Env,
    stats: Optional[Tuple[np.ndarray, np.ndarray]],
    episodes: int,
    deterministic: bool,
) -> Tuple[List[float], List[float]]:
    """ejecutar episodios de evaluacion y coleccionar recompensas y scores."""
    rewards: List[float] = []
    scores: List[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=Config.SEED + episode)
        done = False
        total_reward = 0.0
        episode_score = 0.0
        while not done:
            obs_norm = normalize_obs(obs, stats)
            action, _, _ = agent.get_action(obs_norm, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            # capturar el score acumulado del episodio (Flappy Bird lo provee en info)
            if 'score' in info:
                episode_score = float(info['score'])
        rewards.append(total_reward)
        scores.append(episode_score)
    return rewards, scores


def evaluate_configuration(
    rank: int,
    row: pd.Series,
    config_dir: Path,
    device: str,
    episodes: int,
    deterministic: bool,
) -> Optional[EvalResult]:
    """cargar pesos + stats para una configuracion y calcular metricas de evaluacion."""
    model_path = locate_model_file(config_dir)
    if model_path is None:
        print(f"[WARN] No .pth checkpoint found for {config_dir.name}")
        return None
    stats_path = locate_stats_file(config_dir)
    stats = load_obs_stats(stats_path)
    normalization_mode = "running_stats" if stats is not None else "scale_400"

    env = create_env()
    try:
        agent = build_agent(env, hidden_size=int(row["hidden_size"]), device=device)
        agent.load(str(model_path))
        agent.actor_critic.eval()

        rewards, scores = run_rollouts(
            agent=agent,
            env=env,
            stats=stats,
            episodes=episodes,
            deterministic=deterministic,
        )
    finally:
        env.close()

    if not rewards:
        print(f"[WARN] No rewards collected for {config_dir.name}")
        return None

    rewards_array = np.asarray(rewards, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32)
    
    return EvalResult(
        rank=rank,
        config_id=int(row["config_id"]),
        config_name=str(row["config_name"]),
        model_path=str(model_path),
        stats_path=str(stats_path) if stats_path else "",
        hidden_size=int(row["hidden_size"]),
        learning_rate=float(row["learning_rate"]),
        entropy_coef=float(row["entropy_coef"]),
        n_epochs=int(row["n_epochs"]),
        batch_size=int(row["batch_size"]),
        episodes=len(rewards),
        mean_reward=float(rewards_array.mean()),
        std_reward=float(rewards_array.std(ddof=0)),
        min_reward=float(rewards_array.min()),
        max_reward=float(rewards_array.max()),
        mean_score=float(scores_array.mean()),
        normalization=normalization_mode,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.allow_stochastic:
        deterministic = False
    else:
        deterministic = args.deterministic

    saved_models_dir = args.saved_models_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranking_df, missing = analyzer.analyze_configs(
        saved_models_dir=saved_models_dir,
        max_config_id=args.max_config_id,
        rolling_window=ROLLING_WINDOW,
    )
    if ranking_df.empty:
        print("[ERROR] No configurations available for evaluation.")
        if missing:
            for issue in missing:
                print(f"  - {issue}")
        return 1
    if missing:
        print("Advertencias durante el análisis:")
        for issue in missing:
            print(f"  - {issue}")

    eval_results: List[EvalResult] = []
    top_k = min(args.top_k, len(ranking_df))
    print(f"[INFO] Evaluating top {top_k} configurations on Flappy Bird...")

    for rank in range(1, top_k + 1):
        row = ranking_df.iloc[rank - 1]
        config_id = int(row["config_id"])
        config_dir = analyzer.discover_config_dir(saved_models_dir, config_id)
        if config_dir is None:
            print(f"[WARN] Config directory missing for id {config_id}")
            continue
        print(f"\n[INFO] ({rank}/{top_k}) Evaluating config {config_id} - {row['config_name']}")
        result = evaluate_configuration(
            rank=rank,
            row=row,
            config_dir=config_dir,
            device=args.device,
            episodes=args.episodes,
            deterministic=deterministic,
        )
        if result:
            eval_results.append(result)

    if not eval_results:
        print("[ERROR] All evaluations failed.")
        return 1

    df = pd.DataFrame([res.to_record() for res in eval_results])
    
    # re-ordenar por mean_score (pipes) ya que es la métrica más interpretable
    df = df.sort_values(by=['mean_score'], ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1  # actualizar ranks segun score
    
    df.to_csv(output_path, index=False)

    print("\nRESULTADOS EVALUACIÓN TOP MODELOS")
    print("(Ordenados por SCORE - Pipes Atravesados)")
    print("=" * 80)
    columns = [
        "rank",
        "config_id",
        "config_name",
        "mean_score",      # score primero
        "mean_reward",
        "std_reward",
        "max_reward",
    ]
    print(df.loc[:, columns].to_string(index=False, float_format="%.2f"))
    
    print("\n" + "=" * 80)
    print("NOTAS:")
    print("  - mean_score: Promedio de pipes atravesados (objetivo del juego)")
    print("  - mean_reward: Reward acumulado (0.1/frame + 1.0/pipe - penalizaciones)")
    print("  - Ranking basado en SCORE para interpretabilidad")
    print(f"\n✓ Resultados guardados en: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

