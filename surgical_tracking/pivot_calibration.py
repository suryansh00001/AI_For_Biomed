"""
Surgical Stylus Pivot Calibration.
Calibrates the exact 3D tool tip offset vector (t_tip) by pivoting
the stylus around a stationary point (divot) in 3D space.
Mathematical formulation:
  R_k * t_tip + T_k = P_divot  for k = 1...N
  [R_k  -I] * [t_tip; P_divot] = -T_k
"""
import numpy as np
from typing import List, Tuple, Dict, Optional


class PivotCalibrator:
    def __init__(self, min_samples: int = 40):
        self.min_samples = min_samples
        self.rotation_matrices: List[np.ndarray] = []
        self.translations: List[np.ndarray] = []
        self.is_calibrating = False

    def start(self):
        """Reset and start collecting calibration poses."""
        self.rotation_matrices.clear()
        self.translations.clear()
        self.is_calibrating = True

    def add_sample(self, R: np.ndarray, tvec: np.ndarray) -> int:
        """Adds an orientation sample if currently calibrating."""
        if not self.is_calibrating:
            return 0
        self.rotation_matrices.append(R.copy())
        self.translations.append(tvec.flatten().copy())
        return len(self.rotation_matrices)

    def stop_and_solve(self) -> Tuple[bool, Optional[Dict]]:
        """
        Solves least-squares system for tool tip offset and divot position.
        Returns:
          success: bool
          result: {
            "tip_offset": np.ndarray (3,),
            "divot_pos": np.ndarray (3,),
            "rmse_mm": float,
            "sample_count": int
          }
        """
        self.is_calibrating = False
        n = len(self.rotation_matrices)
        if n < self.min_samples:
            return False, {
                "error": f"Insufficient samples: {n}/{self.min_samples}. Pivot through more angles."
            }

        # Build A matrix (3*N, 6) and b vector (3*N,)
        A = np.zeros((3 * n, 6), dtype=np.float64)
        b = np.zeros((3 * n,), dtype=np.float64)
        I3 = np.eye(3, dtype=np.float64)

        for i in range(n):
            R = self.rotation_matrices[i]
            T = self.translations[i]
            row = i * 3
            A[row:row + 3, 0:3] = R
            A[row:row + 3, 3:6] = -I3
            b[row:row + 3] = -T

        # Solve Ax = b
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        tip_offset = x[0:3]
        divot_pos = x[3:6]

        # Compute root mean square error (RMSE)
        errors = []
        for i in range(n):
            predicted_divot = self.rotation_matrices[i] @ tip_offset + self.translations[i]
            err = np.linalg.norm(predicted_divot - divot_pos)
            errors.append(err)
            
        rmse = float(np.sqrt(np.mean(np.square(errors))))

        return True, {
            "tip_offset": tip_offset,
            "divot_pos": divot_pos,
            "rmse_mm": rmse,
            "sample_count": n
        }

