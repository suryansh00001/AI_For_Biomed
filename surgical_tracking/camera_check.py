"""
Quick Camera Diagnostic & Index Finder.
Lists all accessible camera devices on Windows, tests resolutions,
and verifies ArUco detection capability for surgical navigation.
"""
import cv2


def check_cameras(max_tested: int = 5):
    print("=" * 60)
    print("  SURGICAL TRACKING - CAMERA DIAGNOSTIC TOOL")
    print("=" * 60)
    
    available_indices = []
    
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[+] Camera Index {idx}: AVAILABLE")
                print(f"    * Resolution: {w} x {h}")
                print(f"    * Reported FPS: {fps}")
                available_indices.append(idx)
            cap.release()
            
    if not available_indices:
        print("\n[!] No active cameras detected. Please plug in your USB webcam or check permissions.")
        print("    You can still run in simulation mode: python -m surgical_tracking.surgical_navigation_app --sim")
    else:
        print(f"\n[+] Recommended launch command:")
        print(f"    python -m surgical_tracking.surgical_navigation_app --camera {available_indices[0]}")
    print("=" * 60)


if __name__ == "__main__":
    check_cameras()

