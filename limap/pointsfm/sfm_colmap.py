from _limap import _base, _pointsfm
from .read_write_model import *

def compute_neighbors(model, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou"):
    if neighbor_type == "iou":
        neighbors = model.GetMaxIoUImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "overlap":
        neighbors = model.GetMaxOverlappingImages(n_neighbors, min_triangulation_angle)
    elif neighbor_type == "dice":
        neighbors = model.GetMaxDiceCoeffImages(n_neighbors, min_triangulation_angle)
    else:
        raise NotImplementedError
    return neighbors

def ComputeNeighbors(colmap_output_path, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou", sparse_dir="sparse", images_dir="images"):
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_output_path, sparse_dir, images_dir)

    # get neighbors
    neighbors = compute_neighbors(model, n_neighbors, min_triangulation_angle=min_triangulation_angle, neighbor_type=neighbor_type)
    image_names = model.GetImageNames()
    return neighbors, image_names

# compute neighborhood for a image list sorted as 'image{0:08d}.png'
def ComputeNeighborsSorted(colmap_output_path, n_neighbors, min_triangulation_angle=1.0, neighbor_type="iou", sparse_dir="sparse", images_dir="images"):
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_output_path, sparse_dir, images_dir)

    # get neighbors
    image_names = model.GetImageNames()
    image_id_list = [int(name[5:-4]) for name in image_names]
    neighbors = compute_neighbors(model, n_neighbors, min_triangulation_angle=min_triangulation_angle, neighbor_type=neighbor_type)

    # map indexes
    n1 = [neighbors[image_id_list.index(k)] for k in range(len(image_id_list))]
    n2 = [[image_id_list[val] for val in neighbor] for neighbor in n1]
    neighbors = n2
    return neighbors

def ComputeRanges(colmap_output_path, range_robust=[0.01, 0.99], k_stretch=1.25, sparse_dir="sparse", images_dir="images"):
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_output_path, sparse_dir, images_dir)
    ranges = model.ComputeRanges(range_robust, k_stretch)
    return ranges

def get_camera_info(camera, max_image_dim=None, check_undistorted=True):
    # get original K
    K = np.zeros((3, 3))
    K[2, 2] = 1.0
    dist_coeffs = np.zeros((8))
    if camera.model == "SIMPLE_PINHOLE":
        K[0, 0] = K[1, 1] = camera.params[0]
        K[0, 2], K[1, 2] = camera.params[1], camera.params[2]
    elif camera.model == "PINHOLE":
        K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
        K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
    elif check_undistorted:
        raise ValueError("Image (model = {0}) should be undistorted.".format(camera.model))
    elif camera.model == "SIMPLE_RADIAL":
        K[0, 0] = K[1, 1] = camera.params[0]
        K[0, 2], K[1, 2] = camera.params[1], camera.params[2]
        dist_coeffs[0] = camera.params[3]
    elif camera.model == "RADIAL":
        K[0, 0] = K[1, 1] = camera.params[0]
        K[0, 2], K[1, 2] = camera.params[1], camera.params[2]
        dist_coeffs[0] = camera.params[3]
        dist_coeffs[1] = camera.params[4]
    elif camera.model == "OPENCV":
        K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
        K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
        for i in range(4):
            dist_coeffs = camera.params[i + 4]
    elif camera.model == "FULL_OPENCV":
        K[0, 0], K[1, 1] = camera.params[0], camera.params[1]
        K[0, 2], K[1, 2] = camera.params[2], camera.params[3]
        for i in range(8):
            dist_coeffs[i] = camera.params[i + 4]
    else:
        raise NotImplementedError

    # reshape w.r.t max_image_dim
    h_orig, w_orig = camera.height, camera.width
    img_hw = (h_orig, w_orig)
    if (max_image_dim is not None) and max_image_dim != -1:
        ratio = max_image_dim / max(h_orig, w_orig)
        if ratio < 1.0:
            h_new = int(round(h_orig * ratio))
            w_new = int(round(w_orig * ratio))
            K[0,:] = K[0,:] * w_new / w_orig
            K[1,:] = K[1,:] * h_new / h_orig
            img_hw = (h_new, w_new)
    return K, dist_coeffs, img_hw

def ReadInfosFromModel(model, colmap_path, model_path="sparse", image_path="images", max_image_dim=None, check_undistorted=True):
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

    n_images = len(colmap_images)
    cam_dict = {}
    for image_id, colmap_image in colmap_images.items():
        imname = colmap_image.name
        cam_id = colmap_image.camera_id
        colmap_cam = colmap_cameras[cam_id]
        K, dist_coeffs, img_hw = get_camera_info(colmap_cam, max_image_dim=max_image_dim, check_undistorted=check_undistorted)
        R = qvec2rotmat(colmap_image.qvec)
        T = colmap_image.tvec
        cam = _base.Camera(K, R, T, dist_coeffs)
        cam.set_hw(img_hw[0], img_hw[1])
        cam_dict[imname] = [cam_id, cam]
    imname_list, cameras, cam_id_list = [], [], []
    for imname in image_names:
        imname_list.append(os.path.join(image_path, imname))
        cam_id_list.append(cam_dict[imname][0])
        cameras.append(cam_dict[imname][1])
    return imname_list, cameras, cam_id_list

def ReadInfos(colmap_path, model_path="sparse", image_path="images", max_image_dim=None, check_undistorted=True):
    model = _pointsfm.SfmModel()
    model.ReadFromCOLMAP(colmap_path, model_path, image_path)
    return ReadInfosFromModel(model, colmap_path, model_path=model_path, image_path=image_path, max_image_dim=max_image_dim, check_undistorted=check_undistorted)


