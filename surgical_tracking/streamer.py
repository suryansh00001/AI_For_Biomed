"""
Real-time UDP/Network Telemetry Streamer for Surgical Tool 6-DOF Poses.
Broadcasts tip coordinates, orientation, and margin status at 60 FPS
to Three.js Web Workstation, Unity 3D Bridge, or external medical robotics displays.
"""
import socket
import json
import numpy as np
from typing import Dict, Optional


class SurgicalTelemetryStreamer:
    def __init__(self, host: str = "127.0.0.1", port: int = 9090):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def broadcast_pose(self,
                       tip_pos_cam: np.ndarray,
                       euler_deg: np.ndarray,
                       margins: Optional[Dict[str, float]],
                       patient_id: str,
                       faces_tracked: int):
        """Sends lightweight JSON packet via UDP."""
        packet = {
            "patient_id": patient_id,
            "tip_mm": [round(float(v), 2) for v in tip_pos_cam],
            "rot_deg": [round(float(v), 2) for v in euler_deg],
            "margins_mm": {k: round(float(v), 2) for k, v in (margins or {}).items()},
            "faces_tracked": faces_tracked
        }
        
        try:
            payload = json.dumps(packet).encode("utf-8")
            self.sock.sendto(payload, (self.host, self.port))
        except Exception:
            pass

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

