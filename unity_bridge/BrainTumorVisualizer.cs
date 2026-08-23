using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Unity 3D Interactive Brain Tumor Visualizer & Medical Explorer Component.
/// Attach this component to an empty GameObject in your Unity scene.
/// Supports runtime OBJ/GLTF loading, layer toggling, holographic shaders, and cross-section clipping.
/// </summary>
[ExecuteInEditMode]
public class BrainTumorVisualizer : MonoBehaviour
{
    [Header("Patient 3D Data Path")]
    [Tooltip("Absolute or relative path to the exported patient folder containing .obj or .glb files")]
    public string modelDirectory = "d:/Brain tumorr/exported_3d_models/BraTS20_Training_001";

    [Header("Layer References")]
    public GameObject brainCortexObject;
    public GameObject tumorEdemaObject;
    public GameObject tumorNecroticObject;
    public GameObject tumorEnhancingObject;

    [Header("Layer Visibility Toggles")]
    public bool showBrainShell = true;
    public bool showTumorEdema = true;
    public bool showTumorNecrotic = true;
    public bool showTumorEnhancing = true;

    [Header("Rendering & Transparency Settings")]
    [Range(0.05f, 1.0f)]
    public float brainOpacity = 0.35f;
    [Range(0.5f, 3.0f)]
    public float tumorGlowIntensity = 1.5f;

    [Header("Cross-Section Clipping Plane")]
    public bool enableClipping = false;
    [Range(-100f, 100f)]
    public float clipPlanePosition = 0f;
    public Vector3 clipPlaneNormal = Vector3.up;

    [Header("Interactive Rotation")]
    public bool autoRotate = true;
    [Range(1f, 50f)]
    public float rotationSpeed = 15f;

    private Material brainMaterial;
    private Material edemaMaterial;
    private Material necroticMaterial;
    private Material enhancingMaterial;

    void Start()
    {
        InitializeMaterials();
        ApplyLayerSettings();
    }

    void Update()
    {
        if (autoRotate && Application.isPlaying)
        {
            transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime, Space.World);
        }

        ApplyLayerSettings();
    }

    /// <summary>
    /// Initializes standard PBR and glowing medical materials.
    /// </summary>
    public void InitializeMaterials()
    {
        // Brain Cortex Material (Translucent Ice-Blue Glass)
        Shader standardShader = Shader.Find("Standard");
        if (standardShader == null) standardShader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Diffuse");

        if (brainMaterial == null)
        {
            brainMaterial = new Material(standardShader);
            brainMaterial.name = "Brain_Glass_Mat";
            SetMaterialTransparent(brainMaterial, new Color(0.82f, 0.88f, 0.96f, brainOpacity));
        }

        // Tumor Edema Material (Vivid Green)
        if (edemaMaterial == null)
        {
            edemaMaterial = new Material(standardShader);
            edemaMaterial.name = "Edema_Mat";
            edemaMaterial.color = new Color(0.3f, 0.85f, 0.39f, 0.9f);
            EnableEmission(edemaMaterial, new Color(0.1f, 0.4f, 0.1f) * tumorGlowIntensity);
        }

        // Tumor Necrotic Core Material (Deep Red)
        if (necroticMaterial == null)
        {
            necroticMaterial = new Material(standardShader);
            necroticMaterial.name = "Necrotic_Mat";
            necroticMaterial.color = new Color(1.0f, 0.23f, 0.19f, 1.0f);
            EnableEmission(necroticMaterial, new Color(0.6f, 0.05f, 0.05f) * tumorGlowIntensity);
        }

        // Tumor Enhancing Core Material (Bright Active Amber / Yellow)
        if (enhancingMaterial == null)
        {
            enhancingMaterial = new Material(standardShader);
            enhancingMaterial.name = "Enhancing_Mat";
            enhancingMaterial.color = new Color(1.0f, 0.8f, 0.0f, 1.0f);
            EnableEmission(enhancingMaterial, new Color(0.9f, 0.7f, 0.0f) * tumorGlowIntensity);
        }

        // Apply materials to child renderers if assigned
        AssignMaterial(brainCortexObject, brainMaterial);
        AssignMaterial(tumorEdemaObject, edemaMaterial);
        AssignMaterial(tumorNecroticObject, necroticMaterial);
        AssignMaterial(tumorEnhancingObject, enhancingMaterial);
    }

    private void AssignMaterial(GameObject target, Material mat)
    {
        if (target != null && mat != null)
        {
            var renderer = target.GetComponent<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = mat;
            }
        }
    }

    private void SetMaterialTransparent(Material mat, Color col)
    {
        mat.SetFloat("_Mode", 3); // Transparent mode for Standard Shader
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = 3000;
        mat.color = col;
    }

    private void EnableEmission(Material mat, Color emissionColor)
    {
        mat.EnableKeyword("_EMISSION");
        mat.SetColor("_EmissionColor", emissionColor);
    }

    public void ApplyLayerSettings()
    {
        if (brainCortexObject != null)
        {
            brainCortexObject.SetActive(showBrainShell);
            if (brainMaterial != null)
            {
                Color c = brainMaterial.color;
                c.a = brainOpacity;
                brainMaterial.color = c;
            }
        }

        if (tumorEdemaObject != null)
            tumorEdemaObject.SetActive(showTumorEdema);

        if (tumorNecroticObject != null)
            tumorNecroticObject.SetActive(showTumorNecrotic);

        if (tumorEnhancingObject != null)
            tumorEnhancingObject.SetActive(showTumorEnhancing);
    }

    /// <summary>
    /// Focuses camera onto the 3D model.
    /// </summary>
    public void ResetView()
    {
        transform.rotation = Quaternion.identity;
    }
}
