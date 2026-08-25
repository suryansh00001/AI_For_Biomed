import os
import json
import trimesh
import numpy as np

# Standard Clinical Material Colors (RGBA)
LAYER_MATERIALS = {
    'brain_cortex': {
        'name': 'BrainCortex_Mat',
        'color': [210, 225, 245, 110],      # Semi-transparent pale blue/white
        'metallic': 0.1,
        'roughness': 0.4,
    },
    'tumor_edema': {
        'name': 'TumorEdema_Mat',
        'color': [76, 217, 100, 220],       # Vivid Medical Green
        'metallic': 0.2,
        'roughness': 0.3,
    },
    'tumor_necrotic': {
        'name': 'TumorNecrotic_Mat',
        'color': [255, 59, 48, 255],        # Deep Warning Red
        'metallic': 0.1,
        'roughness': 0.5,
    },
    'tumor_enhancing': {
        'name': 'TumorEnhancing_Mat',
        'color': [255, 204, 0, 255],       # Bright Amber / Yellow (active rim)
        'metallic': 0.3,
        'roughness': 0.2,
    }
}

class MeshExporter:
    """
    Exports 3D brain and tumor meshes to Unity-compatible formats (.OBJ, .GLTF/.GLB, .STL).
    """
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "exported_3d_models"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def export_patient_scene(self, patient_id, meshes_dict, suffix=""):
        """
        Exports all reconstructed layers for a patient into a dedicated directory.
        Supports custom suffix (e.g. '_model', '_gt') for dual AI/Ground-Truth meshes.
        """
        patient_folder = os.path.join(self.output_dir, patient_id)
        os.makedirs(patient_folder, exist_ok=True)

        exported_files = {}
        scene_elements = []
        metadata = {
            'patient_id': patient_id,
            'suffix': suffix,
            'layers': {},
            'exported_files': []
        }

        # 1. Export individual OBJ and STL files with vertex colors
        for layer_name, mesh in meshes_dict.items():
            if mesh is None or len(mesh.vertices) == 0:
                continue

            mat_info = LAYER_MATERIALS.get(layer_name, {
                'name': f"{layer_name}_Mat",
                'color': [200, 200, 200, 255]
            })

            # Assign vertex colors & visual material
            rgba = np.array(mat_info['color'], dtype=np.uint8)
            mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
            mesh.metadata['name'] = layer_name

            # Decimation can drop normals; without them WebGL/Unity shade the
            # surface black because lighting has no surface direction.
            if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
                mesh.compute_vertex_normals()

            # Export individual OBJ
            obj_name = f"{layer_name}{suffix}.obj" if suffix else f"{layer_name}.obj"
            obj_path = os.path.join(patient_folder, obj_name)
            mesh.export(obj_path, file_type='obj')

            # Export individual STL (for 3D printing / engineering)
            stl_name = f"{layer_name}{suffix}.stl" if suffix else f"{layer_name}.stl"
            stl_path = os.path.join(patient_folder, stl_name)
            mesh.export(stl_path, file_type='stl')

            exported_files[layer_name] = {
                'obj': obj_path,
                'stl': stl_path,
                'vertex_count': int(len(mesh.vertices)),
                'face_count': int(len(mesh.faces)),
                'volume_cm3': float(abs(mesh.volume) / 1000.0) if mesh.is_watertight else None
            }

            metadata['layers'][layer_name] = exported_files[layer_name]
            scene_elements.append(mesh)

        # 2. Export single composite GLB / GLTF
        if scene_elements:
            scene = trimesh.Scene(scene_elements)
            glb_filename = f"{patient_id}{suffix}_composite.glb" if suffix else f"{patient_id}_composite.glb"
            glb_path = os.path.join(patient_folder, glb_filename)
            scene.export(glb_path, file_type='glb')
            exported_files['composite_glb'] = glb_path
            metadata['exported_files'].append(glb_path)

        # 3. Save metadata manifest
        manifest_filename = f"manifest{suffix}.json" if suffix else "manifest.json"
        manifest_path = os.path.join(patient_folder, manifest_filename)
        with open(manifest_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[MeshExporter] Successfully exported 3D assets to: {patient_folder} (suffix={suffix})")
        return exported_files

if __name__ == "__main__":
    exporter = MeshExporter()
    print("MeshExporter initialized.")
