# Unity 3D Brain & Tumor Visualization Guide

This folder provides a complete workflow for loading and interacting with the 3D reconstructed brain and tumor models inside the **Unity Game Engine** (Standard, URP, or HDRP).

---

## 1. Quick Start in Unity (Drag & Drop)

1. Open your Unity Project.
2. In the Unity Project window, create a new folder named `BrainTumor3D`.
3. Copy the exported patient folder (e.g. `exported_3d_models/BraTS20_Training_001` from this repository) or drag the `.obj` / `.glb` files into your Unity `BrainTumor3D` folder:
   - `brain_cortex.obj`
   - `tumor_edema.obj`
   - `tumor_necrotic.obj`
   - `tumor_enhancing.obj`
   - or `BraTS20_Training_001_composite.glb`
4. Drag all 4 mesh assets into your Unity **Hierarchy** under an empty Parent GameObject named `BrainModel`.

---

## 2. Interactive C# Script Setup

1. Copy [`BrainTumorVisualizer.cs`](file:///d:/Brain%20tumorr/unity_bridge/BrainTumorVisualizer.cs) into your Unity `Assets/Scripts/` folder.
2. Attach the `BrainTumorVisualizer` component to the `BrainModel` GameObject in the Inspector.
3. In the Inspector, assign the 4 child GameObjects to their respective slots:
   - **Brain Cortex Object**: `brain_cortex`
   - **Tumor Edema Object**: `tumor_edema`
   - **Tumor Necrotic Object**: `tumor_necrotic`
   - **Tumor Enhancing Object**: `tumor_enhancing`

---

## 3. Features & Controls in Unity

- **Holographic / Glass Brain Shell**: Control the `Brain Opacity` slider ($0.05$ to $1.0$) to see deep within the brain anatomy.
- **Layer Visibility Toggles**: Check/uncheck `Show Edema`, `Show Necrotic Core`, or `Show Enhancing Tumor` to inspect specific sub-compartments.
- **Glowing Tumor Cores**: Control `Tumor Glow Intensity` for realistic clinical HDR emissive rendering.
- **360° Auto-Rotation**: Enable `Auto Rotate` for medical kiosk / presentation mode.
- **VR / AR Ready**: Compatible with Meta Quest, HTC Vive, Apple Vision Pro, and Microsoft HoloLens using standard XR Interaction Toolkit.
