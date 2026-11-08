# 🔬 Phase 2: 6-DOF Surgical Stylus Tracking & AR Brain Navigation

This module implements an **optical surgical navigation system** (similar to Medtronic StealthStation or Brainlab Cranial) that tracks a multi-faced fiducial stylus (a pen with a multi-marker dice on its back) with **6 Degrees of Freedom (6-DOF)** and projects the 3D reconstructed patient brain in **Augmented Reality (AR)** with live millimeter-accurate surgical margin telemetry.

---

## 🏗 System Architecture & Geometry

```
                    [ 6-FACED FIDUCIAL CUBE ]
                    +-----------------------+
                    |        [TOP]          |
                    | [LEFT] [FRONT] [RIGHT]|  <-- Multi-Marker Pose Fusion
                    |        [BOTTOM]       |      (Continuous 360° roll tracking)
                    +-----------------------+
                                |
                                |  d_pen (e.g. 135 mm)
                                |
                                ▼
                           [ ● TOOL TIP ]      <-- P_tip = R * t_tip + T
                                |
               - - - - - - - - -|- - - - - - - - -
              |                 ▼                 |
              |     [ 🧠 AR BRAIN MODEL ]         |  <-- Pinned above Table Marker #0
              |  - Cortex Wireframe (Holographic) |
              |  - Edema Sub-region (Yellow)      |
              |  - Enhancing Rim (Cyan)           |
              |  - Necrotic Core (Red)            |
              |                                   |
              |    LIVE MARGIN TELEMETRY (mm)     |
               - - - - - - - - - - - - - - - - - -
```

---

## 🚀 Quick Start Guide

### Step 1: Print & Cut the Stylus Net
Generate or grab the printable markers:
```bash
python -m surgical_tracking.marker_generator
```
Open and print:
- **`surgical_tracking/printable_markers/printable_cube_net_template.png`**: Cut along the outline, fold around your dice, and tape/glue it.
- **`surgical_tracking/printable_markers/patient_anchor_id_0.png`**: Place this flat on your desk to act as the "Operating Table" anchor where the 3D Brain spawns.

### Step 2: Attach the Cube to Your Pen
- Attach the dice to the top/back of your pen (the pen shaft extends out from the **BOTTOM** face of the cube).

### Step 3: Launch Live Surgical Navigation
Run with your webcam:
```bash
python -m surgical_tracking.surgical_navigation_app --camera 0
```

#### Custom Dimensions:
If your dice or pen has different dimensions:
```bash
python -m surgical_tracking.surgical_navigation_app --cube-size 30 --marker-size 22 --pen-length 140
```

#### Instant Simulation Mode (No Webcam / Headless):
To test the AR brain overlay, trajectory line, and margin alerts without a camera:
```bash
python -m surgical_tracking.surgical_navigation_app --sim
```

---

## ⌨️ Interactive Controls (In-App)

| Key | Action | Clinical Purpose |
| :---: | :--- | :--- |
| **`P`** | **Pivot Calibration** | Swivel the pen tip in a fixed divot to auto-compute millimeter-exact tip offset. |
| **`C`** | **Toggle Brain Cortex** | Hide/show the holographic outer brain cortex to see internal tumor cores clearly. |
| **`T`** | **Trajectory Laser** | Toggle the forward-projecting surgical trajectory needle guide line. |
| **`Q` / `ESC`** | **Quit** | Close the application. |

---

## 📐 Mathematical Foundation

### 1. Multi-Face Rigid Body Fusion
Each marker $k \in \{1 \dots 6\}$ has 4 predefined 3D corner coordinates $P_{k, i} \in \mathbb{R}^3$ relative to the cube origin $(0, 0, 0)$.
When the camera observes $M$ faces simultaneously ($4M$ total 2D-to-3D correspondences), the joint 6-DOF pose is solved via Levenberg-Marquardt:
$$\min_{\mathbf{R}, \mathbf{T}} \sum_{j=1}^{4M} \left\| p_j - \pi(\mathbf{R} P_j + \mathbf{T}) \right\|^2$$
Because all faces are rigidly linked, as the surgeon rotates the pen, markers enter and exit view seamlessly without losing tracking.

### 2. Tool Tip Computation
The physical tip in the camera coordinate frame is:
$$P_{\text{tip}}^{\text{cam}} = \mathbf{R} \cdot \vec{t}_{\text{tip}} + \mathbf{T}$$
where $\vec{t}_{\text{tip}} = [0, -d_{\text{pen}}, 0]^T$ in the local cube frame.

### 3. Pivot Calibration (Least Squares)
When pivoting the tool around a stationary point $P_{\text{divot}}$ across $N$ orientations:
$$\mathbf{R}_k \vec{t}_{\text{tip}} + \mathbf{T}_k = P_{\text{divot}}$$
$$[\mathbf{R}_k \quad -\mathbf{I}_3] \begin{bmatrix} \vec{t}_{\text{tip}} \\ P_{\text{divot}} \end{bmatrix} = -\mathbf{T}_k$$
Solving this $3N \times 6$ overdetermined system yields the sub-millimeter calibrated tip offset $\vec{t}_{\text{tip}}$ and residual RMSE.

### 4. 1-Euro Adaptive Low-Pass Filter
Eliminates high-frequency camera detection jitter while preserving instantaneous response during rapid surgical maneuvers:
$$\alpha = \frac{1}{1 + \frac{\tau}{\Delta t}}, \quad f_c = f_{c,\min} + \beta \cdot \|\dot{x}\|$$

