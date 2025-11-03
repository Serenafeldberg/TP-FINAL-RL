import numpy as np
from src.envs import preprocess

def test_to_gray_rgb():
    img = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    gray = preprocess.to_gray(img)
    assert gray.shape == (84, 84)
    assert gray.dtype == np.uint8

def test_resize_shape():
    img = np.random.randint(0, 256, (80, 60, 3), dtype=np.uint8)
    resized = preprocess.resize(img, (40, 40))
    assert resized.shape[:2] == (40, 40)
    assert resized.dtype == np.uint8

def test_resize_chw():
    img = np.random.randint(0, 256, (3, 80, 60), dtype=np.uint8)
    resized = preprocess.resize(img, (40, 40))
    assert resized.shape == (3, 40, 40)

def test_normalize_uint8():
    img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    norm = preprocess.normalize_obs(img)
    assert norm.dtype == np.float32
    assert np.all((norm >= 0) & (norm <= 1))

def test_normalize_float():
    obs = np.random.randn(5, 5)
    norm = preprocess.normalize_obs(obs)
    assert np.allclose(norm, obs.astype(np.float32))
