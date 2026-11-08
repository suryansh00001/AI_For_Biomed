"""
One-Euro Filter for 6-DOF Surgical Pose and Tool Tip Stabilization.
Eliminates high-frequency camera detection jitter while maintaining zero latency.
Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in HCI"
"""
import math
import numpy as np
from typing import Optional


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[np.ndarray] = None

    def filter(self, value: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = np.copy(value)
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter:
    """
    Adaptive low-pass filter:
    - min_cutoff: Cutoff frequency at zero speed (smaller = smoother when stationary)
    - beta: Speed coefficient (larger = less lag during fast hand movement)
    - d_cutoff: Derivative cutoff frequency
    """
    def __init__(self,
                 freq: float = 30.0,
                 min_cutoff: float = 1.0,
                 beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None
        self.last_x: Optional[np.ndarray] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        
        if self.last_time is None or timestamp is None:
            dt = 1.0 / self.freq
        else:
            dt = timestamp - self.last_time
            if dt <= 0:
                dt = 1.0 / self.freq
                
        self.last_time = timestamp

        # Compute derivative (speed)
        if self.last_x is None:
            dx = np.zeros_like(x)
        else:
            dx = (x - self.last_x) / dt

        self.last_x = np.copy(x)

        # Filter the derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = self.dx_filter.filter(dx, alpha_d)

        # Adaptive cutoff based on speed magnitude
        speed = np.linalg.norm(edx)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the signal
        alpha = self._alpha(cutoff, dt)
        return self.x_filter.filter(x, alpha)


class Pose6DOFFilter:
    """Smooths 6-DOF pose (tvec and rvec) and tip position."""
    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.04):
        self.tvec_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.rvec_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.tip_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def filter_pose(self, tvec: np.ndarray, rvec: np.ndarray, tip: np.ndarray, timestamp: float):
        smooth_tvec = self.tvec_filter.filter(tvec.flatten(), timestamp).reshape(3, 1)
        smooth_rvec = self.rvec_filter.filter(rvec.flatten(), timestamp).reshape(3, 1)
        smooth_tip = self.tip_filter.filter(tip.flatten(), timestamp)
        return smooth_tvec, smooth_rvec, smooth_tip

