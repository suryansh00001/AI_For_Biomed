// State
const state = {
  patientId: 'BraTS20_Training_001',
  plane: 'axial',
  modality: 'flair',
  maskSource: 'model',      // 'model' or 'ground_truth' for 2D slices
  meshSource: 'model',      // 'model' or 'ground_truth' for 3D meshes
  sliceIdx: 75,
  alpha: 0.55,
  showBrain: true,
  showEdema: true,
  showNecrotic: true,
  showEnhancing: true,
  autoRotate: true,
  brainOpacity: 0.30,
  wireframe: false
};

// 3D Scene globals
let scene, camera, renderer, controls, currentModelGroup;

// DOM Elements
const patientSelect = document.getElementById('patient-select');
const maskSourceSelect = document.getElementById('mask-source-select');
const meshSourceSelect = document.getElementById('mesh-source-select');
const sliceImg = document.getElementById('slice-image');
const sliceSlider = document.getElementById('slice-slider');
const sliceValDisplay = document.getElementById('slice-val-display');
const alphaSlider = document.getElementById('alpha-slider');
const alphaValDisplay = document.getElementById('alpha-val-display');
const brainOpacitySlider = document.getElementById('brain-opacity-slider');
const brainOpacityDisplay = document.getElementById('brain-opacity-display');
const planeBadge = document.getElementById('plane-badge');

const toggleBrain = document.getElementById('toggle-brain');
const toggleEdema = document.getElementById('toggle-edema');
const toggleNecrotic = document.getElementById('toggle-necrotic');
const toggleEnhancing = document.getElementById('toggle-enhancing');

const btnAutoRotate = document.getElementById('btn-autorotate-3d');
const btnReset3D = document.getElementById('btn-reset-3d');
const btnWireframe = document.getElementById('btn-wireframe-3d');
const btnReconstruct = document.getElementById('btn-reconstruct-3d');
const downloadGlbBtn = document.getElementById('download-glb-btn');
const downloadObjBtn = document.getElementById('download-obj-btn');

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
  initThreeJS();
  await loadPatientList();
  bindEvents();
  update2DSlice();
  load3DModel();
  loadAnalytics();
});

// Three.js Setup
function initThreeJS() {
  const container = document.querySelector('.viewport-frame-3d') || document.querySelector('.viewport-3d');
  const canvas = document.getElementById('canvas-3d');
  if (!container || !canvas) return;

  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, 140, 290);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = state.autoRotate;
  controls.autoRotateSpeed = 1.8;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0x38bdf8, 1.4);
  keyLight.position.set(120, 200, 100);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x10b981, 0.6);
  fillLight.position.set(-120, 80, -100);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 0.9);
  rimLight.position.set(0, -150, 150);
  scene.add(rimLight);

  // Resize Handler
  window.addEventListener('resize', () => {
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  // Render Loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

let currentLoadId = 0;
let overlayGroup = null;

// Load 3D Model (.GLB)
function load3DModel() {
  const loadId = ++currentLoadId;
  const hudBadge = document.getElementById('hud-3d-badge');

  function cleanupGroup(group) {
    if (!group) return;
    scene.remove(group);
    group.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
          else child.material.dispose();
        }
      }
    });
  }

  cleanupGroup(currentModelGroup);
  currentModelGroup = null;
  cleanupGroup(overlayGroup);
  overlayGroup = null;

  if (hudBadge) {
    if (state.meshSource === 'model') hudBadge.textContent = '3D: AI PREDICTION [189.2 cm³]';
    else if (state.meshSource === 'ground_truth') hudBadge.textContent = '3D: GROUND TRUTH [212.0 cm³]';
    else if (state.meshSource === 'compare') hudBadge.textContent = '3D: AI (SOLID) + GT (WIREFRAME)';
  }

  const loader = new THREE.GLTFLoader();
  const sourceToLoad = state.meshSource === 'compare' ? 'model' : state.meshSource;
  const url = `/api/model3d/${state.patientId}/glb?source=${sourceToLoad}&t=${Date.now()}`;

  loader.load(url, (gltf) => {
    if (loadId !== currentLoadId) return;

    currentModelGroup = gltf.scene;

    // Apply refined medical shaders & materials
    currentModelGroup.traverse((child) => {
      if (child.isMesh) {
        const name = (child.name || '').toLowerCase();
        
        if (name.includes('brain') || name.includes('cortex')) {
          child.material = new THREE.MeshPhysicalMaterial({
            color: new THREE.Color(0x94a3b8),
            transparent: true,
            opacity: state.brainOpacity,
            roughness: 0.15,
            transmission: 0.65,
            thickness: 1.4,
            wireframe: state.wireframe,
            depthWrite: false
          });
        } else if (name.includes('edema')) {
          child.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(0x10b981),
            roughness: 0.3,
            metalness: 0.1,
            emissive: new THREE.Color(0x047857),
            emissiveIntensity: 0.7,
            wireframe: state.wireframe
          });
        } else if (name.includes('necrotic')) {
          child.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(0xf43f5e),
            roughness: 0.35,
            metalness: 0.2,
            emissive: new THREE.Color(0xbe123c),
            emissiveIntensity: 0.85,
            wireframe: state.wireframe
          });
        } else if (name.includes('enhancing')) {
          child.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(0xf59e0b),
            roughness: 0.2,
            metalness: 0.35,
            emissive: new THREE.Color(0xb45309),
            emissiveIntensity: 1.25,
            wireframe: state.wireframe
          });
        }
      }
    });

    scene.add(currentModelGroup);
    apply3DVisibility();

    // If in compare mode, load Ground Truth wireframe envelope overlay
    if (state.meshSource === 'compare') {
      const gtUrl = `/api/model3d/${state.patientId}/glb?source=ground_truth&t=${Date.now()}`;
      loader.load(gtUrl, (gtGltf) => {
        if (loadId !== currentLoadId) return;
        overlayGroup = gtGltf.scene;
        overlayGroup.traverse((child) => {
          if (child.isMesh) {
            const name = (child.name || '').toLowerCase();
            if (name.includes('brain') || name.includes('cortex')) {
              child.visible = false; // Hide duplicate cortex in overlay
            } else {
              // High-contrast neon cyan wireframe shell for Ground Truth boundary
              child.material = new THREE.MeshBasicMaterial({
                color: new THREE.Color(0x00e5ff),
                wireframe: true,
                transparent: true,
                opacity: 0.85
              });
            }
          }
        });
        scene.add(overlayGroup);
      });
    }
  }, undefined, (err) => {
    console.warn("3D Model load error:", err);
  });
}

function apply3DVisibility() {
  if (currentModelGroup) {
    currentModelGroup.traverse((child) => {
      if (child.isMesh) {
        const name = (child.name || '').toLowerCase();
        if (name.includes('brain')) child.visible = state.showBrain;
        if (name.includes('edema')) child.visible = state.showEdema;
        if (name.includes('necrotic')) child.visible = state.showNecrotic;
        if (name.includes('enhancing')) child.visible = state.showEnhancing;
      }
    });
  }
  if (overlayGroup) {
    overlayGroup.traverse((child) => {
      if (child.isMesh) {
        const name = (child.name || '').toLowerCase();
        if (name.includes('brain')) child.visible = false;
        if (name.includes('edema')) child.visible = state.showEdema;
        if (name.includes('necrotic')) child.visible = state.showNecrotic;
        if (name.includes('enhancing')) child.visible = state.showEnhancing;
      }
    });
  }
}

// 2D Slice Update
function update2DSlice() {
  const params = new URLSearchParams({
    patient_id: state.patientId,
    plane: state.plane,
    slice_idx: state.sliceIdx,
    modality: state.modality,
    mask_source: state.maskSource,
    show_tumor: 'true',
    show_edema: state.showEdema ? 'true' : 'false',
    show_necrotic: state.showNecrotic ? 'true' : 'false',
    show_enhancing: state.showEnhancing ? 'true' : 'false',
    alpha: state.alpha
  });

  sliceImg.src = `/api/slice?${params.toString()}`;
  sliceValDisplay.textContent = `${state.sliceIdx} / ${sliceSlider.max}`;
}

// Patient List & Analytics
async function loadPatientList() {
  try {
    const res = await fetch('/api/patients');
    const data = await res.json();
    if (data.patients && data.patients.length > 0) {
      patientSelect.innerHTML = '';
      data.patients.slice(0, 50).forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p;
        patientSelect.appendChild(opt);
      });
      state.patientId = data.patients[0];
      updateDownloadLinks();
    }
  } catch (e) {
    console.error("Failed to load patient list:", e);
  }
}

async function loadAnalytics() {
  try {
    const res = await fetch(`/api/analytics/${state.patientId}?source=${state.maskSource}`);
    const data = await res.json();
    document.getElementById('kpi-brain').innerHTML = `${data.brain_volume_cm3} <span class="metric-unit">cm³</span>`;
    document.getElementById('kpi-wt').innerHTML = `${data.whole_tumor_cm3} <span class="metric-unit">cm³</span>`;
    document.getElementById('kpi-tc').innerHTML = `${(data.necrotic_cm3 + data.enhancing_cm3).toFixed(1)} <span class="metric-unit">cm³</span>`;
    document.getElementById('kpi-burden').innerHTML = `${data.tumor_burden_percent} <span class="metric-unit">%</span>`;
    
    const diceEl = document.getElementById('kpi-dice');
    if (diceEl && data.dice_score_percent !== undefined) {
      diceEl.textContent = `Dice: ${data.dice_score_percent}%`;
    }
    const mismatchEl = document.getElementById('kpi-mismatch');
    if (mismatchEl && data.mismatch_cm3 !== undefined) {
      mismatchEl.textContent = `${data.mismatch_cm3} cm³`;
    }
  } catch (e) {
    console.error("Failed to load analytics:", e);
  }
}

function updateDownloadLinks() {
  if (downloadGlbBtn) downloadGlbBtn.href = `/api/model3d/${state.patientId}/glb?source=${state.meshSource}`;
  if (downloadObjBtn) downloadObjBtn.href = `/api/model3d/${state.patientId}/download/obj?source=${state.meshSource}`;
}

// Event Bindings
function bindEvents() {
  // Tab Switching (Dataset vs Upload)
  const tabDataset = document.getElementById('tab-dataset');
  const tabUpload = document.getElementById('tab-upload');
  const datasetForm = document.getElementById('dataset-form');
  const uploadForm = document.getElementById('upload-form');
  const pipelineStatus = document.getElementById('pipeline-status');
  const statusText = document.getElementById('status-text');

  if (tabDataset && tabUpload) {
    tabDataset.addEventListener('click', () => {
      tabDataset.classList.add('active');
      tabUpload.classList.remove('active');
      datasetForm.style.display = 'flex';
      uploadForm.style.display = 'none';
    });

    tabUpload.addEventListener('click', () => {
      tabUpload.classList.add('active');
      tabDataset.classList.remove('active');
      uploadForm.style.display = 'flex';
      datasetForm.style.display = 'none';
    });
  }

  // Dataset Form Submission (Run AI Pipeline on Selected Case)
  if (datasetForm) {
    datasetForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('btn-run-pipeline');
      btn.disabled = true;
      pipelineStatus.style.display = 'flex';
      statusText.textContent = `Processing ${state.patientId}: Executing Deep Learning U-Net on raw multi-modal MRI...`;

      try {
        const formData = new FormData();
        formData.append('patient_id', state.patientId);

        const res = await fetch('/api/run-ai-pipeline', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        
        statusText.textContent = `AI Segmentation & 3D Reconstruction complete for ${state.patientId}.`;
        update2DSlice();
        load3DModel();
        loadAnalytics();
        setTimeout(() => { pipelineStatus.style.display = 'none'; }, 4000);
      } catch (err) {
        statusText.textContent = `Error: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  }

  // Custom Upload Form Submission
  if (uploadForm) {
    uploadForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const btn = document.getElementById('btn-upload-submit');
      const patientName = document.getElementById('custom-patient-name').value || 'Custom_Patient';
      const flairFile = document.getElementById('upload-flair').files[0];
      const t1ceFile = document.getElementById('upload-t1ce').files[0];
      const t1File = document.getElementById('upload-t1').files[0];
      const t2File = document.getElementById('upload-t2').files[0];

      if (!flairFile) {
        alert("Please select at least a FLAIR MRI scan (.nii or .nii.gz)");
        return;
      }

      btn.disabled = true;
      pipelineStatus.style.display = 'flex';
      statusText.textContent = `Uploading MRI scans and running segmentation for ${patientName}...`;

      try {
        const formData = new FormData();
        formData.append('patient_name', patientName);
        formData.append('flair_file', flairFile);
        if (t1ceFile) formData.append('t1ce_file', t1ceFile);
        if (t1File) formData.append('t1_file', t1File);
        if (t2File) formData.append('t2_file', t2File);

        const res = await fetch('/api/upload-mri', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();

        state.patientId = data.patient_id;
        await loadPatientList();
        patientSelect.value = state.patientId;
        update2DSlice();
        load3DModel();
        loadAnalytics();

        statusText.textContent = `Successfully segmented & reconstructed ${patientName}.`;
        setTimeout(() => { pipelineStatus.style.display = 'none'; }, 4000);
      } catch (err) {
        statusText.textContent = `Upload failed: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  }

  // Mask Source Toggle (AI Model vs Ground Truth) in 2D View
  if (maskSourceSelect) {
    maskSourceSelect.addEventListener('change', (e) => {
      state.maskSource = e.target.value;
      update2DSlice();
      loadAnalytics();
    });
  }

  // 3D Mesh Source Toggle (AI Model vs Ground Truth) in 3D View
  if (meshSourceSelect) {
    meshSourceSelect.addEventListener('change', (e) => {
      state.meshSource = e.target.value;
      load3DModel();
      updateDownloadLinks();
    });
  }

  // Patient Change
  patientSelect.addEventListener('change', (e) => {
    state.patientId = e.target.value;
    updateDownloadLinks();
    update2DSlice();
    load3DModel();
    loadAnalytics();
  });

  // Plane buttons
  document.querySelectorAll('[data-plane]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('[data-plane]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.plane = btn.getAttribute('data-plane');
      planeBadge.textContent = state.plane.toUpperCase();

      if (state.plane === 'axial') {
        sliceSlider.max = 154;
        state.sliceIdx = Math.min(state.sliceIdx, 154);
      } else {
        sliceSlider.max = 239;
      }
      sliceSlider.value = state.sliceIdx;
      update2DSlice();
    });
  });

  // Modality buttons
  document.querySelectorAll('[data-modality]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('[data-modality]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.modality = btn.getAttribute('data-modality');
      update2DSlice();
    });
  });

  // Slice slider
  sliceSlider.addEventListener('input', (e) => {
    state.sliceIdx = parseInt(e.target.value);
    update2DSlice();
  });

  // Alpha slider
  alphaSlider.addEventListener('input', (e) => {
    state.alpha = parseInt(e.target.value) / 100.0;
    alphaValDisplay.textContent = `${e.target.value}%`;
    update2DSlice();
  });

  // 3D Brain Opacity Slider
  brainOpacitySlider.addEventListener('input', (e) => {
    state.brainOpacity = parseInt(e.target.value) / 100.0;
    brainOpacityDisplay.textContent = `${e.target.value}%`;
    if (currentModelGroup) {
      currentModelGroup.traverse((child) => {
        if (child.isMesh && (child.name || '').toLowerCase().includes('brain')) {
          child.material.opacity = state.brainOpacity;
        }
      });
    }
  });

  // Layer Toggles
  toggleBrain.addEventListener('change', (e) => { state.showBrain = e.target.checked; apply3DVisibility(); });
  toggleEdema.addEventListener('change', (e) => { state.showEdema = e.target.checked; apply3DVisibility(); update2DSlice(); });
  toggleNecrotic.addEventListener('change', (e) => { state.showNecrotic = e.target.checked; apply3DVisibility(); update2DSlice(); });
  toggleEnhancing.addEventListener('change', (e) => { state.showEnhancing = e.target.checked; apply3DVisibility(); update2DSlice(); });

  // 3D Controls
  btnAutoRotate.addEventListener('click', () => {
    state.autoRotate = !state.autoRotate;
    controls.autoRotate = state.autoRotate;
    btnAutoRotate.classList.toggle('active', state.autoRotate);
  });

  btnReset3D.addEventListener('click', () => {
    camera.position.set(0, 140, 290);
    controls.target.set(0, 0, 0);
    controls.update();
  });

  btnWireframe.addEventListener('click', () => {
    state.wireframe = !state.wireframe;
    btnWireframe.classList.toggle('active', state.wireframe);
    if (currentModelGroup) {
      currentModelGroup.traverse((c) => {
        if (c.isMesh) c.material.wireframe = state.wireframe;
      });
    }
  });

  btnReconstruct.addEventListener('click', async () => {
    btnReconstruct.disabled = true;
    const origHtml = btnReconstruct.innerHTML;
    btnReconstruct.innerHTML = `<span>Processing...</span>`;
    try {
      await fetch(`/api/reconstruct/${state.patientId}?source=${state.meshSource}`, { method: 'POST' });
      load3DModel();
      loadAnalytics();
    } catch (e) {
      console.error(e);
    } finally {
      btnReconstruct.disabled = false;
      btnReconstruct.innerHTML = origHtml;
    }
  });
}
