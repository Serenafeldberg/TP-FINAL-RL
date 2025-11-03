import numpy as np
from src.envs.wrappers import make_env, ActionRepeat, FrameStack, RewardClip, PreprocessObs
import gymnasium as gym

def test_action_repeat():
    env = ActionRepeat(gym.make("CarRacing-v3"), repeat=3)
    obs, info = env.reset()
    a = env.action_space.sample()
    obs2, rew, done, info = env.step(a)
    assert isinstance(rew, float)
    assert rew != 0 or done or obs2 is not None

def test_preprocess_obs_gray_resize_normalize():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = PreprocessObs(env, gray=True, resize_shape=(84,84), normalize=True)
    obs, info = env.reset()
    assert obs.shape == (84,84)
    assert obs.dtype == np.float32

def test_framestack_basic():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = PreprocessObs(env, gray=True, resize_shape=(84,84), normalize=True)
    env = FrameStack(env, k=4)
    obs, info = env.reset()
    assert obs.shape[0] == 4  # stacked frames
    obs2, rew, done, info = env.step(env.action_space.sample())
    assert obs2.shape[0] == 4

def test_make_env_chain():
    env = make_env("CarRacing-v3", seed=0, frame_stack=4, action_repeat=2, gray=True, resize_shape=(84,84))
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.ndim >= 2
