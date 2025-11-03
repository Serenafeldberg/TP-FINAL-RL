import os
import torch

class Config:
    #ENVIRONMENT
    ENV_NAME = "ALE/Skiing-v5"
    RENDER_MODE = None
    SEED = 42

    #PREPROCESSING
    ACTION_REPEAT = 4
    FRAME_STACK = 4
    FRAME_SIZE = 84
    GRAYSCALE = True
    NORMALIZE_OBS = True
    CLIP_REWARDS = True
    MAX_EPISODE_STEPS = None

    #PPO HYPERPARAMETERS
    LEARNING_RATE = 2.5e-4 #OpenAI Baselines para Atari
    LR_DECAY = True
    CLIP_EPSILON = 0.2 #PPO Paper (Schulman 2017)
    GAMMA = 0.99 
    GAE_LAMBDA = 0.95 #GAE Paper (Schulman 2016)
    N_STEPS = 128 #Stable-Baselines3 Atari
    N_ENVS = 1
    BATCH_SIZE = 256
    N_EPOCHS = 4
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    #TRAINING
    TOTAL_TIMESTEPS = 1_000_000
    SAVE_FREQ = 50_000
    LOG_FREQ = 2048
    EVAL_FREQ = 25_000
    N_EVAL_EPISODES = 10

    #MODEL ARCHITECTURE
    CNN_CHANNELS = [32, 64, 64]
    CNN_KERNELS = [8, 4, 3]
    CNN_STRIDES = [4, 2, 1]
    HIDDEN_SIZE = 512

    #PATHS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    MODEL_DIR = os.path.join(BASE_DIR, "savedModels")
    VIDEO_DIR = os.path.join(LOG_DIR, "videos")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    #DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("CONFIGURACIÓN PPO - SKIING")
        print("=" * 60)
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and key.isupper():
                print(f"{key:25s}: {value}")
        print("=" * 60)

    @classmethod
    def get_env_args(cls):
        return dict(
            frame_stack=cls.FRAME_STACK,
            action_repeat=cls.ACTION_REPEAT,
            clip_rewards=cls.CLIP_REWARDS,
            max_episode_steps=cls.MAX_EPISODE_STEPS,
            gray=cls.GRAYSCALE,
            resize_shape=(cls.FRAME_SIZE, cls.FRAME_SIZE),
            normalize_obs=cls.NORMALIZE_OBS,
        )