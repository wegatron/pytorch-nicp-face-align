import bpy
import bmesh
import numpy as np
import sys


# Set the path to your OBJ file
input_obj = sys.argv[4]
input_lm = sys.argv[5]


#input_obj = 'FFHQ-UV/examples/pan_multiview_test1/unwarp/pan/pan_front.obj'
#input_lm = 'output/landmarks/pan/align_lm.txt'

print('input_obj: ' + input_obj)
print('input_lm: ' + input_lm)

# Import the OBJ file
bpy.ops.import_scene.obj(filepath=input_obj, use_edges=True, use_smooth_groups=True, use_split_objects=False, use_split_groups=False, use_groups_as_vgroups=True, use_image_search=True, split_mode='OFF', global_clamp_size=0.0, axis_forward='-Z', axis_up='Y')

# Optional: Center the imported object in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode = 'EDIT')

# Deselect all vertices
bpy.ops.mesh.select_all(action='DESELECT')

me = obj.data
bm = bmesh.from_edit_mesh(me)
bm.verts.ensure_lookup_table()

# visualize landmarks
lm = np.loadtxt(input_lm)[:, 1].astype(int)

for v in bm.verts:
    v.select = False
    
for ind in lm:
    bm.verts[ind].select = True

# update highlight the selected vertex
bmesh.update_edit_mesh(me)
