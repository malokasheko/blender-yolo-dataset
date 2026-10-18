import bpy, os, random, math, glob
from bpy_extras.object_utils import world_to_camera_view

# ================= CONFIG =================
MODEL_PATH   = r"C:\blender_task\free_porsche_911_carrera_4s.glb"
TEXTURE_DIR  = r"C:\blender_task\textures\textures"
OUT_DIR      = r"C:\blender_task\output_dataset"
NUM_IMAGES   = 50
REAL_SIZE_M  = 4.55     # car length in meters
ALTITUDE_M   = 45.0
LENS_MM      = 100
SENSOR_MM    = 36
RES          = 1280
PLANE_SIZE_M = 100.0
TILE_SIZE_M  = 4.0      # real-world size of one texture tile
CLASS_ID     = 0
SEED         = 42
# ==========================================

random.seed(SEED)
img_dir = os.path.join(OUT_DIR, "images")
lbl_dir = os.path.join(OUT_DIR, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

# ---- clean scene ----
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# ---- import model ----
ext = os.path.splitext(MODEL_PATH)[1].lower()
if ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=MODEL_PATH)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=MODEL_PATH)
elif ext == ".obj":
    try:
        bpy.ops.wm.obj_import(filepath=MODEL_PATH)       # Blender 4.x
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=MODEL_PATH)    # Blender 3.x
else:
    raise ValueError("Unsupported format")

# join all meshes into one object
meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
bpy.ops.object.select_all(action='DESELECT')
for m in meshes:
    m.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.join()
obj = bpy.context.active_object
obj.rotation_mode = 'XYZ'          # glTF imports use quaternion; this makes rotation_euler work
obj.parent = None
obj.matrix_world.translation = (0, 0, 0)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# rescale to real-world size, center origin on bounding box
s = REAL_SIZE_M / max(obj.dimensions)
obj.scale = (s, s, s)
bpy.ops.object.transform_apply(scale=True)
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
obj.rotation_euler = (0, 0, 0)
HALF_H = obj.dimensions.z / 2      # keeps the object sitting on the ground

# ---- ground plane with PBR ----
bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE_M, location=(0, 0, 0))
ground = bpy.context.active_object
mat = bpy.data.materials.new("Ground")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
tc = nt.nodes.new("ShaderNodeTexCoord")
mp = nt.nodes.new("ShaderNodeMapping")
k = PLANE_SIZE_M / TILE_SIZE_M
mp.inputs["Scale"].default_value = (k, k, k)
nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])

def tex(pattern, noncolor):
    files = glob.glob(os.path.join(TEXTURE_DIR, pattern))
    if not files:
        return None
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(files[0])
    if noncolor:
        n.image.colorspace_settings.name = "Non-Color"
    nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
    return n

d = tex("*diff*", False)
r = tex("*rough*", True)
nrm = tex("*nor_gl*", True)
print("Textures found -> diffuse:", d is not None,
      "rough:", r is not None, "normal:", nrm is not None)
if d:   nt.links.new(d.outputs["Color"], bsdf.inputs["Base Color"])
if r:   nt.links.new(r.outputs["Color"], bsdf.inputs["Roughness"])
if nrm:
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
    nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
ground.data.materials.append(mat)

# ---- camera: straight down ----
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = LENS_MM
cam_data.sensor_width = SENSOR_MM
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (0, 0, ALTITUDE_M)
cam.rotation_euler = (0, 0, 0)     # looks along -Z
bpy.context.scene.camera = cam

half_fp = ALTITUDE_M * SENSOR_MM / (2 * LENS_MM)   # half the ground width seen (~8.1 m at 100 mm)
limit = half_fp - REAL_SIZE_M * 0.6                # margin so the car stays fully in frame
print(f"Footprint: {2*half_fp:.1f} m, placement limit: +/-{limit:.2f} m")

# ---- sun + render settings ----
sun_data = bpy.data.lights.new("Sun", 'SUN')
sun = bpy.data.objects.new("Sun", sun_data)
bpy.context.collection.objects.link(sun)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.image_settings.file_format = 'JPEG'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.exposure = 1.0
scene.world = scene.world or bpy.data.worlds.new("World")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 3.0

def yolo_bbox(o):
    dg = bpy.context.evaluated_depsgraph_get()
    eo = o.evaluated_get(dg)
    me = eo.to_mesh()
    pts = [world_to_camera_view(scene, cam, o.matrix_world @ v.co) for v in me.vertices]
    eo.to_mesh_clear()
    xs = [min(max(p.x, 0), 1) for p in pts]
    ys = [min(max(p.y, 0), 1) for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    # image y-axis is flipped relative to camera view
    return ((x0 + x1) / 2, 1 - (y0 + y1) / 2, x1 - x0, y1 - y0)

# ---- generation loop ----
for i in range(NUM_IMAGES):
    obj.location = (random.uniform(-limit, limit),
                    random.uniform(-limit, limit),
                    HALF_H)
    obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))
    sun.rotation_euler = (math.radians(random.uniform(15, 50)),
                          0,
                          math.radians(random.uniform(0, 360)))
    sun_data.energy = random.uniform(5, 10)
    bpy.context.view_layer.update()

    cx, cy, w, h = yolo_bbox(obj)
    name = f"img_{i:04d}"
    scene.render.filepath = os.path.join(img_dir, name + ".jpg")
    bpy.ops.render.render(write_still=True)
    with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
        f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    print(f"Rendered {i+1}/{NUM_IMAGES}")

with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
    f.write("car\n")
print("Done")