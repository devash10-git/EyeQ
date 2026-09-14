"""Gaze feature specification and dataset synthesis for the quantum pipeline.

The classical front end (``static/iris-tracker.js``) already extracts an
8-dimensional geometric vector per frame:

    [1, hL, hR, vL, vR, noseX, noseY, faceW]

where hL/hR and vL/vR are iris positions expressed as *ratios inside the eye
opening* (scale- and translation-invariant by construction), and the nose /
inter-corner-width terms encode head translation and distance.

This module extends that to the 13-dimensional research feature vector used by
the quantum modules, and synthesises datasets from the *same forward model* the
tracker assumes, so that simulation results are grounded in the real geometry
rather than in arbitrary blobs.

IMPORTANT — provenance of data
------------------------------
No recorded human gaze corpus ships with this project. Every dataset produced
here is **synthetic**, generated from the documented forward model below. It is
therefore valid for *relative* comparison of algorithms under a stated noise
model, and is explicitly NOT evidence of real-world accuracy. Any published
claim must repeat these experiments on recorded human data.

Forward model (the physics the tracker linearises)
--------------------------------------------------
    screen_x = a0 + a1*h + a2*(h - 0.5)^3 + a3*(noseX - 0.5) + eps
    screen_y = b0 + b1*v + b2*(v - 0.5)^3 + b3*(noseY - 0.5) + eps

The cubic term is the true optical curvature; the head terms are the
compensation the 8-feature vector exists to provide. `eps` is landmark jitter,
whose magnitude is the dominant error source in a webcam setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Feature specification
# --------------------------------------------------------------------------

#: Ordered names of the 13-dimensional research feature vector.
FEATURE_NAMES: Tuple[str, ...] = (
    "iris_x",        # horizontal iris position within the eye opening   [0,1]
    "iris_y",        # vertical iris position within the eyelids         [0,1]
    "ear",           # eye aspect ratio (openness)                       [0,1]
    "blink",         # blink indicator / probability                     [0,1]
    "yaw",           # head yaw                                          [-1,1]
    "pitch",         # head pitch                                        [-1,1]
    "roll",          # head roll                                         [-1,1]
    "eye_openness",  # normalised palpebral aperture                     [0,1]
    "gaze_velocity", # |d(gaze)/dt|, normalised                          [0,1]
    "saccade",       # saccade indicator                                 [0,1]
    "fixation",      # fixation stability score                          [0,1]
    "confidence",    # tracker confidence / frame quality                [0,1]
    "history",       # EMA of recent gaze displacement (movement memory) [0,1]
)

N_FEATURES: int = len(FEATURE_NAMES)

#: The 8-dim subset the *deployed* classical tracker actually computes. Used
#: for the like-for-like comparison against the shipped ridge-regression model.
TRACKER_FEATURE_IDX: Tuple[int, ...] = (0, 1, 2, 4, 5, 7, 10, 11)


@dataclass
class GazeSample:
    """One frame of gaze evidence plus its supervision targets."""

    features: np.ndarray                 # shape (N_FEATURES,)
    screen: Tuple[float, float]          # normalised gaze point in [0,1]^2
    key_index: int = -1                  # index into the keyboard layout
    movement_state: int = 0              # 0=fixation 1=saccade 2=blink 3=drift

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=float)
        if self.features.shape != (N_FEATURES,):
            raise ValueError(
                f"expected {N_FEATURES} features, got {self.features.shape}"
            )


# --------------------------------------------------------------------------
# Normalisation — required before angle encoding
# --------------------------------------------------------------------------

class FeatureScaler:
    """Min-max scaler mapping features into ``[0, pi]``.

    Angle encoding writes a feature onto a rotation angle. Values must occupy a
    bounded, non-wrapping interval or two distinct inputs alias onto the same
    quantum state: ``RY(0)`` and ``RY(2*pi)`` are physically identical. ``[0, pi]``
    is the standard injective choice for a single rotation axis.
    """

    def __init__(self, out_max: float = np.pi) -> None:
        self.out_max = float(out_max)
        self.lo: np.ndarray | None = None
        self.hi: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=float)
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lo is None or self.hi is None:
            raise RuntimeError("FeatureScaler.fit must be called first")
        X = np.asarray(X, dtype=float)
        span = np.where(self.hi - self.lo < 1e-12, 1.0, self.hi - self.lo)
        return np.clip((X - self.lo) / span, 0.0, 1.0) * self.out_max

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# Keyboard geometry — mirrors static/index.html
# --------------------------------------------------------------------------

QWERTY_ROWS: Tuple[str, ...] = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def keyboard_layout() -> List[Tuple[str, float, float, float, float]]:
    """Return ``(letter, x, y, w, h)`` rectangles in normalised screen space.

    Geometry follows the deployed keyboard: three staggered letter rows filling
    the lower portion of the viewport.
    """
    layout: List[Tuple[str, float, float, float, float]] = []
    row_h = 0.22
    y0 = 0.30
    for r, row in enumerate(QWERTY_ROWS):
        n = len(row)
        w = 1.0 / len(QWERTY_ROWS[0])          # key width set by the longest row
        indent = (1.0 - n * w) / 2.0           # centre shorter rows
        for i, ch in enumerate(row):
            layout.append((ch, indent + i * w, y0 + r * row_h, w, row_h))
    return layout


def key_at(x: float, y: float) -> int:
    """Index of the key containing ``(x, y)``, or ``-1``. Mirrors hit-testing."""
    for i, (_ch, kx, ky, kw, kh) in enumerate(keyboard_layout()):
        if kx <= x < kx + kw and ky <= y < ky + kh:
            return i
    return -1


# --------------------------------------------------------------------------
# Dataset synthesis
# --------------------------------------------------------------------------

def _forward(h: float, v: float, nose_x: float, nose_y: float,
             curvature: float) -> Tuple[float, float]:
    """Ground-truth gaze->screen map, including optical curvature + head shift."""
    sx = 0.5 + 1.05 * (h - 0.5) + curvature * (h - 0.5) ** 3 + 0.35 * (nose_x - 0.5)
    sy = 0.5 + 1.05 * (v - 0.5) + curvature * (v - 0.5) ** 3 + 0.35 * (nose_y - 0.5)
    return float(np.clip(sx, 0.0, 1.0)), float(np.clip(sy, 0.0, 1.0))


def make_gaze_dataset(
    n_samples: int = 600,
    noise: float = 0.02,
    curvature: float = 0.45,
    head_motion: float = 0.06,
    seed: int = 13,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a key-classification dataset.

    Args:
        n_samples: number of frames.
        noise: landmark jitter std-dev, in eye-ratio units. 0.02 is a realistic
            webcam figure; the sweep in ``benchmark.py`` varies it.
        curvature: strength of the cubic optical term (0 = perfectly linear).
        head_motion: std-dev of head translation during the session.
        seed: RNG seed.

    Returns:
        ``(X, y_key, screen)`` with ``X`` of shape ``(n, N_FEATURES)``,
        ``y_key`` the target key index, and ``screen`` the true gaze points.
    """
    rng = np.random.default_rng(seed)
    layout = keyboard_layout()

    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)
    pts = np.zeros((n_samples, 2))

    for i in range(n_samples):
        # target a key uniformly, then look at a point inside it
        k = rng.integers(0, len(layout))
        _ch, kx, ky, kw, kh = layout[k]
        tx = kx + kw * rng.uniform(0.25, 0.75)
        ty = ky + kh * rng.uniform(0.25, 0.75)

        nose_x = 0.5 + rng.normal(0, head_motion)
        nose_y = 0.5 + rng.normal(0, head_motion)

        # invert the linear part to find the iris ratios that hit the target
        h = 0.5 + (tx - 0.5 - 0.35 * (nose_x - 0.5)) / 1.05
        v = 0.5 + (ty - 0.5 - 0.35 * (nose_y - 0.5)) / 1.05
        h += rng.normal(0, noise)
        v += rng.normal(0, noise)

        sx, sy = _forward(h, v, nose_x, nose_y, curvature)

        ear = float(np.clip(rng.normal(0.75, 0.06), 0.0, 1.0))
        vel = float(np.clip(abs(rng.normal(0.05, 0.04)), 0.0, 1.0))
        X[i] = [
            h, v, ear, 0.0,
            (nose_x - 0.5) * 2, (nose_y - 0.5) * 2, rng.normal(0, 0.05),
            ear, vel, 0.0,
            float(np.clip(1.0 - vel * 4, 0.0, 1.0)),
            float(np.clip(rng.normal(0.85, 0.08), 0.0, 1.0)),
            float(np.clip(abs(rng.normal(0.1, 0.06)), 0.0, 1.0)),
        ]
        pts[i] = (sx, sy)
        y[i] = key_at(sx, sy)

    keep = y >= 0                    # drop gazes that landed outside the board
    return X[keep], y[keep], pts[keep]


def make_movement_dataset(
    n_samples: int = 600, seed: int = 13
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesise an eye-movement state dataset.

    Four classes with the kinematic signatures used in the eye-tracking
    literature (I-VT style velocity thresholding is the classical baseline):

    * fixation — low velocity, eyes open, high stability
    * saccade  — high velocity, eyes open, low stability
    * blink    — low EAR / eye closure, velocity unreliable
    * drift    — low-but-nonzero velocity, moderate stability
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        cls = int(rng.integers(0, 4))
        if cls == 0:      # fixation
            vel, ear, stab = abs(rng.normal(0.03, 0.02)), rng.normal(0.78, 0.05), rng.normal(0.90, 0.06)
        elif cls == 1:    # saccade
            vel, ear, stab = abs(rng.normal(0.62, 0.16)), rng.normal(0.74, 0.06), rng.normal(0.14, 0.08)
        elif cls == 2:    # blink
            vel, ear, stab = abs(rng.normal(0.30, 0.22)), rng.normal(0.12, 0.07), rng.normal(0.30, 0.16)
        else:             # drift
            vel, ear, stab = abs(rng.normal(0.16, 0.06)), rng.normal(0.76, 0.05), rng.normal(0.58, 0.10)

        vel = float(np.clip(vel, 0, 1)); ear = float(np.clip(ear, 0, 1)); stab = float(np.clip(stab, 0, 1))
        X[i] = [
            rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), ear,
            1.0 if cls == 2 else 0.0,
            rng.normal(0, 0.15), rng.normal(0, 0.15), rng.normal(0, 0.08),
            ear, vel,
            1.0 if cls == 1 else 0.0,
            stab,
            float(np.clip(rng.normal(0.85, 0.08), 0, 1)),
            float(np.clip(vel * 0.8 + rng.normal(0, 0.05), 0, 1)),
        ]
        y[i] = cls
    return X, y


def letter_frequencies(unigrams: dict) -> np.ndarray:
    """Letter unigram frequencies (a-z) from the project's real vocabulary."""
    counts = np.zeros(26)
    for word, c in unigrams.items():
        for ch in word:
            if "a" <= ch <= "z":
                counts[ord(ch) - 97] += c
    total = counts.sum()
    return counts / total if total else counts


def letter_bigrams(unigrams: dict) -> np.ndarray:
    """Letter *transition* counts (26x26), row-normalised, from real vocabulary.

    Drives the keyboard-layout objective: frequently adjacent letters should be
    placed far apart to reduce confusion, and close together to reduce travel —
    the tension QAOA is asked to resolve.
    """
    M = np.zeros((26, 26))
    for word, c in unigrams.items():
        for a, b in zip(word, word[1:]):
            if "a" <= a <= "z" and "a" <= b <= "z":
                M[ord(a) - 97, ord(b) - 97] += c
    row = M.sum(axis=1, keepdims=True)
    return np.divide(M, np.where(row == 0, 1, row))
