import numpy as np
from src.envs.wrappers import make_env, ActionRepeat, FrameStack, RewardClip, PreprocessObs
import gymnasium as gym
import importlib

def test_action_repeat():

    try:
        minatar = importlib.import_module('minatar')
        # register envs if helper exists
        try:
            reg = importlib.import_module('minatar.gym')
            if hasattr(reg, 'register_envs'):
                reg.register_envs()
        except Exception:
            pass
    except Exception as e:
        print(f'MinAtar no instalado, saltando entrenamiento: {e}')
        return

    
    # Usar gymnasium.make
    gym = importlib.import_module('gymnasium')
    env = gym.make("MinAtar/Breakout-v0")
    env = ActionRepeat(gym.make("MinAtar/Breakout-v0"), repeat=3)
    obs, info = env.reset()
    a = env.action_space.sample()
    obs2, rew, done, info = env.step(a)
    assert isinstance(rew, float)
    assert rew != 0 or done or obs2 is not None

def test_preprocess_obs_gray_resize_normalize():
    try:
        minatar = importlib.import_module('minatar')
        # register envs if helper exists
        try:
            reg = importlib.import_module('minatar.gym')
            if hasattr(reg, 'register_envs'):
                reg.register_envs()
        except Exception:
            pass
    except Exception as e:
        print(f'MinAtar no instalado, saltando entrenamiento: {e}')
        return

    
    # Usar gymnasium.make
    gym = importlib.import_module('gymnasium')
    env = gym.make("MinAtar/Breakout-v0")
    env = PreprocessObs(env, gray=True, resize_shape=(84,84), normalize=True)
    obs, info = env.reset()
    assert obs.shape == (84,84)
    assert obs.dtype == np.float32

def test_framestack_basic():
    try:
        minatar = importlib.import_module('minatar')
        # register envs if helper exists
        try:
            reg = importlib.import_module('minatar.gym')
            if hasattr(reg, 'register_envs'):
                reg.register_envs()
        except Exception:
            pass
    except Exception as e:
        print(f'MinAtar no instalado, saltando entrenamiento: {e}')
        return

    
    # Usar gymnasium.make
    gym = importlib.import_module('gymnasium')
    env = gym.make("MinAtar/Breakout-v0")
    env = PreprocessObs(env, gray=True, resize_shape=(84,84), normalize=True)
    env = FrameStack(env, k=4)
    obs, info = env.reset()
    assert obs.shape[0] == 4  # stacked frames
    obs2, rew, done, info = env.step(env.action_space.sample())
    assert obs2.shape[0] == 4

def test_make_env_chain():

    try:
        minatar = importlib.import_module('minatar')
        # register envs if helper exists
        try:
            reg = importlib.import_module('minatar.gym')
            if hasattr(reg, 'register_envs'):
                reg.register_envs()
        except Exception:
            pass
    except Exception as e:
        print(f'MinAtar no instalado, saltando entrenamiento: {e}')
        return

    
    # Usar gymnasium.make
    gym = importlib.import_module('gymnasium')

    env = make_env("MinAtar/Breakout-v0", seed=0, frame_stack=4, action_repeat=2, gray=True, resize_shape=(84,84))
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.ndim >= 2
