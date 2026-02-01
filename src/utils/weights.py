"""
Weight sampling utilities.
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np


def sample_poisson_weights(
    *,
    num_samples: int,
    dim: int,
    lam: float = 1.0,
    seed: Optional[int] = None,
    max_tries: int = 10000,
) -> List[List[float]]:
    """
    Sample `num_samples` distinct preference weight vectors of length `dim`
    using Poisson counts, then normalize to sum to 1.

    - Resamples if all-zero.
    - Tries to ensure uniqueness (up to rounding tolerance).
    """
    if dim <= 0:
        raise ValueError("dim must be > 0")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if lam <= 0:
        raise ValueError("lam must be > 0")

    rng = np.random.default_rng(seed)
    out: List[List[float]] = []
    seen = set()

    tries = 0
    while len(out) < num_samples and tries < max_tries:
        tries += 1
        counts = rng.poisson(lam=lam, size=dim).astype(np.float32)
        s = float(counts.sum())
        if s <= 0:
            continue
        w = counts / s

        # uniqueness via rounding
        key = tuple(np.round(w, 4).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append([float(x) for x in w.tolist()])

    if len(out) < num_samples:
        # fallback: allow duplicates (should be rare)
        while len(out) < num_samples:
            counts = rng.poisson(lam=lam, size=dim).astype(np.float32)
            s = float(counts.sum())
            if s <= 0:
                continue
            w = counts / s
            out.append([float(x) for x in w.tolist()])

    return out

