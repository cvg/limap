from _limap import _base
from .read_write_model import *

def ReadInfos(model, colmap_path, model_path="sparse", image_path="images"):
    print("Start loading COLMAP sparse reconstruction.")
    image_names = model.GetImageNames()
    model_path = os.path.join(colmap_path, model_path)
    image_path = os.path.join(colmap_path, image_path)
    if os.path.exists(os.path.join(model_path, "cameras.bin")):
        fname_cameras = os.path.join(model_path, "cameras.bin")
        fname_images = os.path.join(model_path, "images.bin")
        colmap_cameras = read_cameras_binary(fname_cameras)
        colmap_images = read_images_binary(fname_images)
    elif os.path.exists(os.path.join(model_path, "cameras.txt")):
        fname_cameras = os.path.join(model_path, "cameras.txt")
        fname_images = os.path.join(model_path, "images.txt")
        colmap_cameras = read_cameras_text(fname_cameras)
        colmap_images = read_images_text(fname_images)
    else:
        raise ValueError("Error! The model file does not exist at {0}".format(model_path))
    print("Reconstruction loaded. (n_images = {0})".format(len(colmap_images)))

    # read cameras
    cameras = {}
    for cam_id, colmap_cam in colmap_cameras.items():
        cameras[cam_id] = _base.Camera(colmap_cam.model, colmap_cam.params, cam_id=cam_id, hw=[colmap_cam.height, colmap_cam.width])

    # read images
    n_images = len(colmap_images)
    camimages = {}
    for img_id, colmap_image in colmap_images.items():
        imname = colmap_image.name
        cam_id = colmap_image.camera_id
        pose = _base.CameraPose(colmap_image.qvec, colmap_image.tvec)
        camimage = _base.CameraImage(cam_id, pose, image_name=os.path.join(image_path, imname))
        camimages[img_id] = camimage

    # get image collection
    imagecols = _base.ImageCollection(cameras, camimages)
    return imagecols

def PyReadCOLMAP(colmap_path, model_path=None):
    if model_path is not None:
        model_path = os.path.join(colmap_path, model_path)
    else:
        model_path = colmap_path
    if os.path.exists(os.path.join(model_path, "points3D.bin")):
        fname_cameras = os.path.join(model_path, "cameras.bin")
        fname_points = os.path.join(model_path, "points3D.bin")
        fname_images = os.path.join(model_path, "images.bin")
        colmap_cameras = read_cameras_binary(fname_cameras)
        colmap_images = read_images_binary(fname_images)
        colmap_points = read_points3D_binary(fname_points)
    elif os.path.exists(os.path.join(model_path, "points3D.txt")):
        fname_cameras = os.path.join(model_path, "cameras.txt")
        fname_points = os.path.join(model_path, "points3D.txt")
        fname_images = os.path.join(model_path, "images.txt")
        colmap_cameras = read_cameras_text(fname_cameras)
        colmap_images = read_images_text(fname_images)
        colmap_points = read_points3D_text(fname_points)
    else:
        raise ValueError("Error! The model file does not exist at {0}".format(model_path))
    return colmap_cameras, colmap_images, colmap_points

