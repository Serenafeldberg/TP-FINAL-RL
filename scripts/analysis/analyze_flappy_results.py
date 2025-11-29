#!/usr/bin/env python3
"""
script para resumir los resultados de entrenamiento de PPO para Flappy Bird.

ejecucion:
    python analyze_flappy_results.py \
        --saved-models-dir savedModels/ \
        --max-config-id 18 \
        --output flappy_config_comparison.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SAVED_MODELS_DIR = BASE_DIR / "savedModels"
DEFAULT_OUTPUT_PATH = BASE_DIR / "flappy_config_comparison.csv"
ROLLING_WINDOW = 100


@dataclass
class ConfigSummary:
    """contenedor para las metricas agregadas de una sola configuracion."""

    config_id: int
    config_label: str
    config_name: str
    mean_reward_last: float
    std_reward_last: float
    max_reward_last: float
    max_reward_all: float
    mean_reward_all: float
    total_episodes: int
    total_timesteps: int
    learning_rate: float
    entropy_coef: float
    n_epochs: int
    hidden_size: int
    batch_size: int
    entropy_recent: Optional[float]
    clip_fraction_recent: Optional[float]
    approx_kl_recent: Optional[float]
    final_learning_rate: Optional[float]
    rewards_path: str
    losses_path: str

    def to_record(self) -> Dict[str, object]:
        """retorna una representacion de diccionario que es amigable para pandas."""
        record = asdict(self)
        record["rewards_path"] = str(self.rewards_path)
        record["losses_path"] = str(self.losses_path)
        return record


def discover_config_dir(saved_models_dir: Path, config_id: int) -> Optional[Path]:
    """
    retorna el directorio para una configuracion dada si existe."""
    explicit = saved_models_dir / f"config_{config_id}"
    if explicit.exists():
        return explicit

    matches = sorted(saved_models_dir.glob(f"config_{config_id}_*"))
    return matches[0] if matches else None


def read_config_metadata(config_dir: Path) -> Dict[str, object]:
    """Load ``config.json`` metadata for a configuration."""
    config_path = config_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    required_fields = [
        "id",
        "name",
        "learning_rate",
        "entropy_coef",
        "n_epochs",
        "hidden_size",
        "batch_size",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise KeyError(
            f"Missing required fields {missing} in {config_path}"
        )
    return data


def _load_csv(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {missing_cols}")
    return df


def summarize_rewards(  # noqa: PLR0913
    rewards_df: pd.DataFrame,
    rolling_window: int,
) -> Tuple[float, float, float, float, float, int, int]:
    """
    calcula las metricas de recompensa para una configuracion.

    Returns:
        mean_recent: media de recompensa en los ultimos ``rolling_window`` episodios.
        std_recent: desviacion estandar en los ultimos ``rolling_window`` episodios.
        max_recent: maximo de recompensa en los ultimos ``rolling_window`` episodios.
        max_all: maximo de recompensa en todo el entrenamiento.
        mean_all: media de recompensa en todo el entrenamiento.
        total_episodes: numero de episodios registrados.
        last_timestep: timestep de la ultima fila.
    """
    rewards_df = rewards_df.sort_values("episode").reset_index(drop=True)
    total_episodes = len(rewards_df)
    last_timestep = int(rewards_df["timestep"].iloc[-1]) if total_episodes else 0

    window = min(rolling_window, total_episodes) or 1
    recent_slice = rewards_df.tail(window)

    recent_rewards = recent_slice["reward"]
    mean_recent = float(recent_rewards.mean())
    std_recent = float(recent_rewards.std(ddof=0))
    max_recent = float(recent_rewards.max())
    max_all = float(rewards_df["reward"].max())
    mean_all = float(rewards_df["reward"].mean())

    return (
        mean_recent,
        std_recent,
        max_recent,
        max_all,
        mean_all,
        total_episodes,
        last_timestep,
    )


def summarize_losses(
    losses_df: pd.DataFrame,
    tail: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if losses_df.empty:
        return (None, None, None, None)

    losses_df = losses_df.sort_values("timestep").reset_index(drop=True)
    window = min(tail, len(losses_df)) or 1
    recent = losses_df.tail(window)

    entropy_recent = float((-recent["entropy_loss"]).mean())
    clip_recent = float(recent["clip_fraction"].mean())
    approx_kl_recent = float(recent["approx_kl"].mean())
    final_lr = float(losses_df["learning_rate"].iloc[-1])

    return (entropy_recent, clip_recent, approx_kl_recent, final_lr)


def summarize_configuration(
    config_dir: Path,
    rolling_window: int,
) -> ConfigSummary:
    metadata = read_config_metadata(config_dir)

    rewards_path = config_dir / "logs" / "rewards.csv"
    losses_path = config_dir / "logs" / "losses.csv"

    rewards_df = _load_csv(
        rewards_path,
        expected_columns=["timestep", "episode", "reward", "length"],
    )
    losses_df = _load_csv(
        losses_path,
        expected_columns=[
            "timestep",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "clip_fraction",
            "approx_kl",
            "learning_rate",
        ],
    )

    (
        mean_recent,
        std_recent,
        max_recent,
        max_all,
        mean_all,
        total_episodes,
        last_timestep,
    ) = summarize_rewards(rewards_df, rolling_window=rolling_window)

    (
        entropy_recent,
        clip_recent,
        approx_kl_recent,
        final_lr,
    ) = summarize_losses(losses_df, tail=rolling_window)

    return ConfigSummary(
        config_id=int(metadata["id"]),
        config_label=config_dir.name,
        config_name=str(metadata["name"]),
        mean_reward_last=mean_recent,
        std_reward_last=std_recent,
        max_reward_last=max_recent,
        max_reward_all=max_all,
        mean_reward_all=mean_all,
        total_episodes=total_episodes,
        total_timesteps=last_timestep,
        learning_rate=float(metadata["learning_rate"]),
        entropy_coef=float(metadata.get("entropy_coef", 0.0)),
        n_epochs=int(metadata.get("n_epochs", 1)),
        hidden_size=int(metadata.get("hidden_size", 0)),
        batch_size=int(metadata.get("batch_size", 0)),
        entropy_recent=entropy_recent,
        clip_fraction_recent=clip_recent,
        approx_kl_recent=approx_kl_recent,
        final_learning_rate=final_lr,
        rewards_path=str(rewards_path),
        losses_path=str(losses_path),
    )


def build_ranking_dataframe(
    summaries: Sequence[ConfigSummary],
) -> pd.DataFrame:
    if not summaries:
        return pd.DataFrame()

    df = pd.DataFrame([summary.to_record() for summary in summaries])
    df = df.sort_values(
        by=["mean_reward_last", "max_reward_last"],
        ascending=False,
    ).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def format_table(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    """
    crea una tabla fija con formato para la salida de consola.

    Args:
        df: dataframe ordenado.
        columns: lista ordenada de columnas para mostrar.
    """
    if df.empty:
        return "No configurations were processed."

    display_df = df.loc[:, columns]

    float_cols = {
        "mean_reward_last",
        "std_reward_last",
        "max_reward_last",
    }
    formatted = display_df.copy()
    for col in formatted.columns:
        if col in float_cols:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):.2f}" if pd.notna(x) else "nan"
            )

    if "learning_rate" in formatted.columns:
        formatted["learning_rate"] = formatted["learning_rate"].map(
            lambda x: f"{float(x):.1e}" if pd.notna(x) else "nan"
        )

    col_widths = {
        col: max(len(str(col)), formatted[col].astype(str).map(len).max())
        for col in formatted.columns
    }

    def _format_row(row: pd.Series) -> str:
        return " | ".join(
            str(row[col]).rjust(col_widths[col]) for col in formatted.columns
        )

    header = " | ".join(
        col.upper().rjust(col_widths[col]) for col in formatted.columns
    )
    separator = "-+-".join("-" * col_widths[col] for col in formatted.columns)
    body = "\n".join(_format_row(row) for _, row in formatted.iterrows())
    return "\n".join([header, separator, body])


def analyze_configs(
    saved_models_dir: Path,
    max_config_id: int,
    rolling_window: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate statistics for all configs up to ``max_config_id``."""
    summaries: List[ConfigSummary] = []
    missing: List[str] = []

    for config_id in range(1, max_config_id + 1):
        config_dir = discover_config_dir(saved_models_dir, config_id)
        if config_dir is None:
            missing.append(f"config_{config_id}* (directory not found)")
            continue
        try:
            summaries.append(
                summarize_configuration(config_dir, rolling_window=rolling_window)
            )
        except FileNotFoundError as exc:
            missing.append(f"{config_dir.name}: missing file -> {exc}")
        except (ValueError, KeyError) as exc:
            missing.append(f"{config_dir.name}: {exc}")

    ranking_df = build_ranking_dataframe(summaries)
    return ranking_df, missing


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze PPO Flappy Bird training runs."
    )
    parser.add_argument(
        "--saved-models-dir",
        type=Path,
        default=DEFAULT_SAVED_MODELS_DIR,
        help="Directory containing config_* folders (default: %(default)s)",
    )
    parser.add_argument(
        "--max-config-id",
        type=int,
        default=18,
        help="Highest configuration id to scan (default: %(default)s)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=ROLLING_WINDOW,
        help="Episodes to use for final statistics (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the exported CSV summary (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry-point."""
    args = parse_args(argv)
    saved_models_dir = args.saved_models_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not saved_models_dir.exists():
        print(f"[ERROR] Directory does not exist: {saved_models_dir}")
        return 1

    ranking_df, missing = analyze_configs(
        saved_models_dir=saved_models_dir,
        max_config_id=args.max_config_id,
        rolling_window=args.rolling_window,
    )

    if ranking_df.empty:
        print("[WARN] No configurations were successfully processed.")
        if missing:
            print("\nIssues detected:")
            for issue in missing:
                print(f"  - {issue}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(output_path, index=False)

    columns = [
        "rank",
        "config_id",
        "config_name",
        "mean_reward_last",
        "std_reward_last",
        "max_reward_last",
        "learning_rate",
        "hidden_size",
    ]
    table = format_table(ranking_df, columns=columns)

    print("\nRANKING CONFIGURACIONES - FLAPPY BIRD\n")
    print(table)
    print(f"\nResumen exportado a: {output_path}")

    if missing:
        print("\nAdvertencias:")
        for issue in missing:
            print(f"  - {issue}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

