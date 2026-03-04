from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def extract_success_from_info(info) -> float | None:
    """Best-effort extraction of scalar success from a single-env info dict."""
    if not isinstance(info, dict):
        return None
    for key in ("success", "is_success"):
        if key not in info:
            continue
        val = info[key]
        try:
            return float(val > 0)
        except Exception:
            try:
                return float(np.asarray(val).reshape(-1)[0] > 0)
            except Exception:
                return None
    return None


def obs_to_np_vector(obs) -> np.ndarray:
    """Convert env observation to a single flat float32 feature vector."""
    if isinstance(obs, dict):
        for key in ("state", "observation", "obs"):
            if key in obs:
                return obs_to_np_vector(obs[key])
        parts = [obs_to_np_vector(obs[k]) for k in sorted(obs.keys())]
        return np.concatenate(parts, axis=0).astype(np.float32)

    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.shape[0] == 1:
        return arr[0].reshape(-1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


@dataclass
class ContinualMetricTracker:
    """Track R_{i,j} matrix and derive standard continual-learning metrics."""

    num_tasks: int
    name: str

    def __post_init__(self):
        self.matrix = np.full((self.num_tasks, self.num_tasks), np.nan, dtype=np.float32)
        self.baseline = np.full((self.num_tasks,), np.nan, dtype=np.float32)

    def set_baseline(self, values: Sequence[float]) -> None:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        n = min(self.num_tasks, vals.shape[0])
        self.baseline[:n] = vals[:n]

    def update(self, row: int, values: Sequence[float]) -> dict[str, float]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        n = min(self.num_tasks, vals.shape[0])
        self.matrix[row, :n] = vals[:n]
        return self.metrics_for_row(row)

    def metrics_for_row(self, row: int) -> dict[str, float]:
        out: dict[str, float] = {}
        current = self.matrix[row]
        out[f"continual/{self.name}/acc_seen"] = float(np.nanmean(current[: row + 1]))
        out[f"continual/{self.name}/acc_all"] = float(np.nanmean(current))

        if row > 0:
            diag = np.diag(self.matrix)
            out[f"continual/{self.name}/bwt"] = float(np.nanmean(current[:row] - diag[:row]))
            out[f"continual/{self.name}/forgetting"] = float(
                np.nanmean(np.nanmax(self.matrix[: row + 1, :row], axis=0) - current[:row])
            )

            if np.any(~np.isnan(self.baseline[1 : row + 1])):
                fwt_vals = []
                for j in range(1, row + 1):
                    if np.isnan(self.baseline[j]) or np.isnan(self.matrix[j - 1, j]):
                        continue
                    fwt_vals.append(float(self.matrix[j - 1, j] - self.baseline[j]))
                if fwt_vals:
                    out[f"continual/{self.name}/fwt"] = float(np.mean(fwt_vals))
        return out
