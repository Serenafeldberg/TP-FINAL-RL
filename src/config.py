import os
import torch

class Config:
    # ENVIRONMENT
    # TODO: ajustar ENV_NAME al id real del entorno de Flappy Bird con LIDAR
    ENV_NAME = "FlappyBird-v0"
    SEED = 42

    # PPO HYPERPARAMETERS
    LEARNING_RATE = 1e-4 #3e-4  # Learning rate estándar para MLP
    LR_DECAY = True
    CLIP_EPSILON = 0.2  # PPO Paper (Schulman 2017)
    GAMMA = 0.99 
    GAE_LAMBDA = 0.95  # GAE Paper (Schulman 2016)
    N_STEPS = 1024  # Steps por rollout
    N_ENVS = 1
    BATCH_SIZE = 64  # Debe ser <= N_STEPS * N_ENVS (128)
    N_EPOCHS = 4
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.05 #0.01
    MAX_GRAD_NORM = 0.5

    # TRAINING
    TOTAL_TIMESTEPS = 1_000_000
    SAVE_FREQ = 50_000 
    LOG_FREQ = 2048
    EVAL_FREQ = 25_000
    N_EVAL_EPISODES = 10

    # MODEL ARCHITECTURE (MLP para observaciones vectoriales)
    HIDDEN_SIZE = 256  # Tamaño de capas ocultas de la MLP

    # PATHS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    LOG_DIR = os.path.join(BASE_DIR, "logs", "flappy")
    MODEL_DIR = os.path.join(BASE_DIR, "savedModels", "flappy")
    VIDEO_DIR = os.path.join(LOG_DIR, "videos")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("CONFIGURACIÓN PPO - FLAPPY BIRD")
        print("=" * 60)
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and key.isupper():
                print(f"{key:25s}: {value}")
        print("=" * 60)

    @classmethod
    def get_env_args(cls):
        """Retorna argumentos para crear el entorno Flappy Bird."""
        return dict(
            # Wrappers genéricos opcionales
            clip_rewards=False,  # No clipear rewards por defecto (ajustar según necesidad)
            max_episode_steps=None,  # Sin límite de pasos por defecto
            render_mode="human",
            use_lidar=True
        )
