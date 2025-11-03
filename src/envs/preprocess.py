"""
Preprocessing helpers for environment observations.

Pure, side-effect-free functions used by wrappers:
- to_gray(frame): RGB/RGBA → 1-canal (grayscale), soporta HWC y CHW.
- resize(frame, shape): redimensiona (H,W[,C]) preservando canales.
- normalize_obs(obs): escala arrays enteros a [0,1] (float32). Floats se dejan igual.

Notas:
- Asumimos orden RGB en HWC (el estándar de Gym/Gymnasium).
- Si llega CHW, convertimos a HWC y devolvemos en mismo orden que entra.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    from PIL import Image
    _HAS_CV2 = False


# ---------- helpers internos ----------

def _is_chw(x: np.ndarray) -> bool:
    return x.ndim == 3 and x.shape[0] in (1, 3, 4)

def _to_hwc(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (1, 2, 0))

def _to_chw(x: np.ndarray) -> np.ndarray:
    return np.transpose(x, (2, 0, 1))


# ---------- API público ----------

def to_gray(frame: np.ndarray) -> np.ndarray:
    """
    Convierte un frame RGB/RGBA a gris (1 canal).
    - Acepta HxW, HxWxC (C=3/4) o CxHxW (C=1/3/4).
    - Devuelve HxW (uint8 si entrada es uint8, sino float32).
    """
    if hasattr(frame, "detach"):  # torch tensor
        frame = frame.cpu().numpy()

    arr = np.asarray(frame)
    if arr.ndim == 2:
        # ya es gris: devolver copia segura del dtype original
        return arr.astype(arr.dtype, copy=True)

    # Detectar formato y llevar a HWC
    input_chw = False
    if _is_chw(arr):
        arr = _to_hwc(arr)
        input_chw = True

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported frame shape for to_gray: {frame.shape}")

    # Ignorar alpha si existe
    rgb = arr[..., :3]

    # Convertir a gris con coeficientes ITU-R BT.709 (sRGB)
    if np.issubdtype(rgb.dtype, np.integer):
        f = rgb.astype(np.float32)
        gray = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
        gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    else:
        f = rgb.astype(np.float32)
        gray = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
        # si ya estaba en [0,1], lo preservamos; no forzamos a 0..255
        gray = gray.astype(np.float32)

    # Siempre devolvemos HxW (no CHW)
    return np.ascontiguousarray(gray)


def resize(frame: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Redimensiona a (height, width) preservando # de canales y dtype.
    Acepta HxW, HxWxC o CxHxW.
    """
    if hasattr(frame, "detach"):
        frame = frame.cpu().numpy()

    arr = np.asarray(frame)
    h, w = shape
    input_chw = False

    if _is_chw(arr):
        arr = _to_hwc(arr)
        input_chw = True

    if arr.ndim == 2:
        # HxW
        if _HAS_CV2:
            interp = cv2.INTER_AREA if (arr.shape[0] > h or arr.shape[1] > w) else cv2.INTER_LINEAR
            out = cv2.resize(arr, (w, h), interpolation=interp)
        else:
            im = Image.fromarray(arr.astype(np.uint8) if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.float32))
            out = np.asarray(im.resize((w, h), resample=Image.BILINEAR))
        out = out.astype(arr.dtype)
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        # HxWxC
        if _HAS_CV2:
            interp = cv2.INTER_AREA if (arr.shape[0] > h or arr.shape[1] > w) else cv2.INTER_LINEAR
            out = cv2.resize(arr, (w, h), interpolation=interp)
        else:
            mode = "L" if arr.shape[2] == 1 else "RGB"
            im = Image.fromarray(arr.squeeze(-1).astype(np.uint8), mode="L") if mode == "L" \
                 else Image.fromarray(arr.astype(np.uint8), mode=mode)
            out = np.asarray(im.resize((w, h), resample=Image.BILINEAR))
            if mode == "L":
                out = out[..., None]
        out = out.astype(arr.dtype)
    else:
        raise ValueError(f"Unsupported frame shape for resize: {arr.shape}")

    if input_chw:
        out = _to_chw(out)
    return np.ascontiguousarray(out)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    """
    Normaliza observaciones de imagen enteras a [0,1] (float32).
    - Para dtypes enteros (uint8/uint16/…): divide por su máx. posible.
    - Para floats: sólo castea a float32 (NO reescala ni clipea).
      Esto evita romper estados continuos de control (e.g., MuJoCo).
    """
    if hasattr(obs, "detach"):
        obs = obs.cpu().numpy()

    arr = np.asarray(obs)

    if np.issubdtype(arr.dtype, np.integer):
        # Escalar por el máximo representable del dtype (p.ej. 255 para uint8)
        info = np.iinfo(arr.dtype)
        out = arr.astype(np.float32) / float(info.max)
        return np.clip(out, 0.0, 1.0)  # por seguridad
    else:
        # Ya es float: conservar escala original, sólo asegurar float32
        return arr.astype(np.float32)
