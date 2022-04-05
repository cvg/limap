from mathutils import *
import argparse
import numpy
import bpy
import glob
import os
import re
import math

# Input: P 3x4 numpy matrix
# Output: K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
#
# Reference implementations: 
#   - Oxford's visual geometry group matlab toolbox 
#   - Scilab Image Processing toolbox
def KRT_from_P(P):
    N = 3
    H = P[:,0:N]  # if not numpy,  H = P.to_3x3()

    [K,R] = rf_rq(H)

    K /= K[-1,-1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = numpy.diag(numpy.sign(numpy.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert; the proj equation remains same:
    if (numpy.linalg.det(R) < 0):
       R = -R
    # C = -H\P[:,-1]
    C = numpy.linalg.lstsq(-H, P[:,-1], rcond=None)[0]
    T = -R*C
    return K, R, T

# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations: 
#   Oxford's visual geometry group matlab toolbox 
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = numpy.linalg.qr(P[ ::-1, ::-1], 'complete')
    q = q.T
    q = q[ ::-1, ::-1]
    r = r.T
    r = r[ ::-1, ::-1]

    if (numpy.linalg.det(q) < 0):
        r[:,0] *= -1
        q[0,:] *= -1
    return r, q

# Creates a blender camera consistent with a given 3x4 computer vision P matrix
# Run this in Object Mode
# scale: resolution scale percentage as in GUI, known a priori
# P: numpy 3x4
def get_blender_camera_from_3x4_P(P, scale, name=None):
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(numpy.matrix(P))

    scene = bpy.context.scene
    sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
    resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0,0] / s_u
    # recover original resolution
    scene.render.resolution_x = resolution_x_in_px / scale
    scene.render.resolution_y = resolution_y_in_px / scale
    scene.render.resolution_percentage = scale * 100

    # Use this if the projection matrix follows the convention listed in my answer to
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Use this if the projection matrix follows the convention from e.g. the matlab calibration toolbox:
    # R_bcam2cv = Matrix(
    #     ((-1, 0,  0),
    #      (0, 1, 0),
    #      (0, 0, 1)))

    R_cv2world = R_world2cv.T
    rotation =  Matrix(R_cv2world.tolist()) @ R_bcam2cv
    location = -R_cv2world * T_world2cv

    # create a new camera
    bpy.ops.object.add(
        type='CAMERA',
        location=location)
    ob = bpy.context.object
    cam = ob.data
    if name is None:
        ob.name = 'CamFrom3x4PObj'
        cam.name = 'CamFrom3x4P'
    else:
        ob.name = name
        cam.name = name

    # Lens
    cam.type = 'PERSP'
    cam.lens = f_in_mm 
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width  = sensor_width_in_mm
    ob.matrix_world = Matrix.Translation(location) @ rotation.to_4x4()

    #     cam.shift_x = -0.05
    #     cam.shift_y = 0.1
    #     cam.clip_start = 10.0
    #     cam.clip_end = 250.0
    #     empty = bpy.data.objects.new('DofEmpty', None)
    #     empty.location = origin+Vector((0,10,0))
    #     cam.dof_object = empty

    # Display
    cam.show_name = True
    # Make this the current camera
    scene.camera = ob
    #bpy.context.scene.update()
    return ob

def test2():
    P = Matrix([
    [2. ,  0. , - 10. ,   282.  ],
    [0. ,- 3. , - 14. ,   417.  ],
    [0. ,  0. , - 1.  , - 18.   ]
    ])
    # This test P was constructed as k*[r | t] where
    #     k = [2 0 10; 0 3 14; 0 0 1]
    #     r = [1 0 0; 0 -1 0; 0 0 -1]
    #     t = [231 223 -18]
    # k, r, t = KRT_from_P(numpy.matrix(P))
    get_blender_camera_from_3x4_P(P, 1)
    

if __name__ == '__main__':    
    calib_dir = bpy.path.abspath(r'//gendarmenmarkt/colmap/')
    view_path_pattern = calib_dir + r'{:03d}.png'
    calib_paths = list(glob.glob(os.path.join(calib_dir, 'calib_*.txt')))
    num_views = len(calib_paths)
    print("Found", num_views, "views")
    cam_objs = []
    for i, path in enumerate(calib_paths):
        print('Importing view {} of {}'.format(i+1, num_views))
        view_id = re.match(r'.*calib_(\d+)\.txt$', path).group(1)
        calib_mat = Matrix(numpy.loadtxt(path))
        cam_obj = get_blender_camera_from_3x4_P(calib_mat, 1, name='View {}'.format(view_id))
        cam_objs.append(cam_obj)
        
        view_path = view_path_pattern.format(int(view_id))
        # Needs "Import Images as Planes" plugin activated
        bpy.ops.import_image.to_plane(shader='SHADELESS', offset=False, align_axis='Z+', files=[{'name': view_path}])
        img_obj = bpy.context.active_object
        scale = 0.6
        #  img_obj.location.z = -2.3 * scale
        img_obj.scale = 3 * [scale]
        img_obj.parent = cam_obj
    
    for cam_obj in cam_objs:
        cam_obj.rotation_mode = 'QUATERNION'
        cam_obj.keyframe_insert('rotation_quaternion', frame=60)
        cam_obj.keyframe_insert('location', frame=60)
        
    
#    width = 4
#    steps = 60
#    for i, cam_obj in enumerate(cam_objs):
#        row = i // width
#        col = i % width
#        cam_obj.location = (col, 0.7 * row, 0)
#        cam_obj.keyframe_insert('location', frame=1)

#        start_quaternion = Quaternion((0, 0, 1), math.pi)
#        end_quaternion = Quaternion(cam_obj.rotation_quaternion)
#        for s in range(steps):
#            fac = s / (steps - 1)
#            quaternion = start_quaternion.slerp(end_quaternion, fac)
#            cam_obj.rotation_quaternion = quaternion
#            cam_obj.keyframe_insert('rotation_quaternion', frame=s+1)