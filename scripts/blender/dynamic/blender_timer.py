import bpy
import os
import numpy as np
from mathutils import Vector

# Data
#basepath = '/local/home/shaoliu/myProjects/stable/limap-internal'
#import_path = os.path.join(basepath, 'tmp/finaltracks')
#timers_path = os.path.join(basepath, 'tmp/timers.npy')
#timers_path = os.path.join(basepath, 'tmp/timers_threshold.npy')

basepath = '/media/shaoliu/cvg-hdd-usb-2tb-08/Lines/experiments/limap_tnt_lsd'
expdir = os.path.join(basepath, 'Truck')
import_path = os.path.join(expdir, 'finaltracks')
timers_path = os.path.join(expdir, 'timers_threshold.npy')

# Some parameters to play with
line_width = 0.02
background_color = [0.025, 0.025, 0.025, 1.0] #RGBA
line_color = [1.0, 1.0, 1.0, 1.0] # RGBA

def read_tracks_limap_folder(folder):
    # linetracks folder
    flist = os.listdir(folder)
    new_flist = []
    for fname in flist:
        if fname[-4:] != '.txt':
            continue
        new_flist.append(fname)
    flist = new_flist
    n_tracks = len(flist)
    lines3D = {}
    for track_id in range(n_tracks):
        fname = 'track_{0}.txt'.format(track_id)
        fname = os.path.join(folder, fname)
        with open(fname, 'r') as f:
            txt_lines = f.readlines()
        arr = txt_lines[0].strip()
        arr = np.array([float(k) for k in arr.split(' ')])
        lines3D[track_id] = {'xyz': arr, 'track': []}
    return lines3D            

def read_timers(fname):
    with open(fname, 'rb') as f:
        timers = np.load(f, allow_pickle=True)
    return timers

def read_tracks(path):
    lines3D = {}
    if not os.path.exists(path):
        return lines3D, images

    with open(path, "r") as fid:
        num_tracks = int(fid.readline().strip())
        for track_k in range(num_tracks):
            line = fid.readline().strip()
            (track_id, n_supporting_segs, n_supporting_images) = [int(x) for x in line.split(' ')]

            line = fid.readline().strip()
            (x1,y1,z1) = [float(x) for x in line.split(' ')]

            line = fid.readline().strip()
            (x2,y2,z2) = [float(x) for x in line.split(' ')]

            line = fid.readline().strip()
            image_ids = [int(x) for x in line.split(' ')]
            line = fid.readline().strip()
            line_ids = [int(x) for x in line.split(' ')]

            lines3D[track_id] = {
                'xyz': np.array([x1,y1,z1,x2,y2,z2]),
                'track': []
            }
    return lines3D

# Create a new collection for all lines
lines_coll = bpy.data.collections.new('Lines')
bpy.context.scene.collection.children.link(lines_coll)

# Create an empty object as origin of the reconstruction.
# This makes it easie to move the line cloud around later
line_origin = bpy.data.objects.new( "line_origin", None )
lines_coll.objects.link(line_origin)
line_origin.empty_display_type = 'PLAIN_AXES'  

#lines3D = read_tracks(import_path)
lines3D = read_tracks_limap_folder(import_path)
timers = read_timers(timers_path)
assert len(lines3D) == timers.shape[0]

# Set the background color
background = bpy.data.worlds['World'].node_tree.nodes['Background']
# Set background color as RGBA
background.inputs['Color'].default_value = background_color

# Create a material for the lines
mat = bpy.data.materials.new(name='LineMat')
mat.use_nodes = True
# Remove the default material
mat.node_tree.nodes.remove(mat.node_tree.nodes["Principled BSDF"])
emission = mat.node_tree.nodes.new('ShaderNodeEmission')
mat_out = mat.node_tree.nodes['Material Output']
emission.inputs['Color'].default_value = line_color
emission.inputs['Strength'].default_value = 1.0

# Link emission shader to material
mat.node_tree.links.new(mat_out.inputs[0], emission.outputs[0])

for line_id, line_data in lines3D.items():
    time = timers[line_id] * 2 + 20
    if timers[line_id] < 0:
        continue
    
    line_coords = line_data['xyz']

    # Read the coordinates
    v1 = [line_coords[0], line_coords[1], line_coords[2]]
    v2 = [line_coords[3], line_coords[4], line_coords[5]]
    
    # Create a path
    crv = bpy.data.curves.new('line', 'CURVE')
    crv.dimensions = '3D'
    spline = crv.splines.new(type='NURBS')
    spline.points.add(1)
    spline.points[0].co = v1 + [1.0]
    spline.points[1].co = v2 + [1.0]
    spline.use_endpoint_u = True
    
    crv.bevel_mode = 'ROUND'
    crv.bevel_depth = line_width 
    crv.use_fill_caps = True
    
    obj = bpy.data.objects.new('line', crv)
    obj.active_material = mat

    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_render", frame=time)
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_render", frame=time+1)
    
    obj.parent = line_origin
    lines_coll.objects.link(obj)


