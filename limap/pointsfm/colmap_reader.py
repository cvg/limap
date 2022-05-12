from _limap import _base
from .read_write_model import *

def ComputeNeighbors(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    if neighbor_type == "iou":
        neighbors = model.GetMaxIoUImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "overlap":
        neighbors = model.GetMaxOverlappingImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "dice":
        neighbors = model.GetMaxDiceCoeffImages(n_neighbors, min_triangulation_angle)
    else:
        raise NotImplementedError
    return neighbors

# compute neighborhood for a image list sorted as 'image{0:08d}.png'
def ComputeNeighborsSorted(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    # get neighbors
    neighbors = ComputeNeighbors(model, n_neighbors, min_triangulation_angle=min_triangulation_angle, neighbor_type=neighbor_type)

    # map indexes
    image_names = model.GetImageNames()
    image_id_list = [int(name[5:-4]) for name in image_names]
    n1 = [neighbors[image_id_list.index(k)] for k in range(len(image_id_list))]
    n2 = [[image_id_list[val] for val in neighbor] for neighbor in n1]
    neighbors = n2
    return neighbors

def ReadInfos(model, colmap_path, model_path="sparse", image_path="images", check_undistorted=True):
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
    camimage_dict = {}
    for image_id, colmap_image in colmap_images.items():
        imname = colmap_image.name
        cam_id = colmap_image.camera_id
        pose = _base.CameraPose(colmap_image.qvec, colmap_image.tvec)
        camimage = _base.CameraImage(cam_id, pose, image_name=os.path.join(image_path, imname))
        camimage_dict[imname] = camimage

    # map to the correct order
    camimages = []
    for imname in image_names:
        camimage = camimage_dict[imname]
        camimages.append(camimage)

    # get image collection
    imagecols = _base.ImageCollection(cameras, camimages)
    return imagecols

