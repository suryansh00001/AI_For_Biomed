# 🎯 Phase 2: Live Demo & Presentation Cheat Sheet

Use this guide when demonstrating the **6-DOF Surgical Tool Tracking & AR Brain Navigation Station** to evaluators, professors, or an audience.

---

## 🎙️ 30-Second Elevator Pitch

> *"In Phase 1, our deep learning Residual U-Net successfully segmented multi-compartment gliomas from multi-modal MRI scans and reconstructed 3D surface meshes.*
> *In Phase 2, we took this into real-time operating theater guidance. We created an optical surgical navigation pointer using a multi-faced fiducial cube. When a surgeon rotates the instrument, single 2D markers fail due to roll occlusion. Our multi-face cube maintains continuous 6 Degrees of Freedom tracking across 360 degrees. We project the patient's actual 3D brain and tumor sub-regions in Augmented Reality, providing live millimeter-accurate surgical margin alarms and trajectory guidance."*

---

## 🔬 Key Engineering Points to Highlight

1. **Why the Cube (Multi-Faced Rigid Body Fusion)?**
   * *Problem*: In optical tracking, tool rolling or steep tilt angles cause single planar markers to vanish.
   * *Solution*: 6 faces rigidly mapped in 3D relative to the cube origin $(0,0,0)$. OpenCV's joint `solvePnP` solver with Levenberg-Marquardt refinement takes all visible 2D-3D correspondences simultaneously. Tracking never drops when twisting or rolling the pen.

2. **Tip Offset Vector & Trajectory Physics**:
   $$P_{\text{tip}}^{\text{cam}} = \mathbf{R} \cdot \vec{t}_{\text{tip}} + \mathbf{T}$$
   $$\vec{v}_{\text{trajectory}} = \mathbf{R} \cdot \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$
   * The green trajectory laser ray guides biopsy needle entry angle into the tumor centroid.

3. **Clinical Pivot Calibration**:
   * Point out the **`P`** key: swiveling the tip in a stationary divot solves the overdetermined least-squares system $[\mathbf{R}_k \quad -\mathbf{I}_3] \begin{bmatrix} \vec{t}_{\text{tip}} \\ P_{\text{divot}} \end{bmatrix} = -\mathbf{T}_k$, determining the exact tool length without manual measurement.

4. **1-Euro Adaptive Filter**:
   * Dynamic cutoff frequency: high smoothing when stationary (eliminating pixel jitter), zero lag during rapid surgical hand maneuvers.

---

## ⌨️ Live Demo Keybindings

| Key | What Happens on Screen | What to Say |
| :---: | :--- | :--- |
| **`N`** | Cycles to next patient (`BraTS20_Training_001` $\rightarrow$ `002` $\rightarrow$ `003`...) | *"We can navigate live across all 23 real patient cases from the BraTS dataset."* |
| **`M`** | Toggles **AI Model Predicted** vs **Ground Truth** | *"Here is our trained U-Net prediction compared directly with the expert radiologist ground truth."* |
| **`C`** | Toggles outer brain cortex wireframe / shell | *"We can hide the outer cortex to inspect deep internal tumor margins clearly."* |
| **`T`** | Toggles surgical trajectory laser line | *"This shows the biopsy trajectory projection extending from the tool tip."* |
| **`P`** | Starts/stops Pivot Calibration | *"Automated calibration of the surgical stylus length."* |
| **`Q`** | Exits application | — |

---

## 🚀 Recommended Demo Flow

1. **Start with Table Anchor**: Place the Anchor Marker (ID 0) on your desk. Show how the 3D Brain model is pinned and locked in space.
2. **Bring in the Tracked Stylus**: Hold the pen in view. Show the RGB 3D axes and virtual pen body following your hand in 6-DOF.
3. **Demonstrate 360° Roll**: Roll the pen between your fingers. Point out that tracking never drops because adjacent faces seamlessly take over.
4. **Approach the Tumor**: Move the tip towards the brain. Point to the HUD on the left showing distance in mm counting down:
   - Green ($> 15\,\text{mm}$): Safe corridor
   - Yellow ($5-15\,\text{mm}$): Approaching margin
   - Red ($< 5\,\text{mm}$): Tumor contact alert
5. **Switch Model Mode**: Press **`M`** to show the AI-predicted tumor boundaries vs ground truth.

