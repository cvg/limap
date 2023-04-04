import os, sys
import time

path = 'training'
scene_id = 'Courthouse'
output_path = 'colmap'
folder_list = os.listdir(path)

for folder in folder_list:
    if folder != scene_id:
        continue
    input_folder = os.path.join(path, folder)
    output_folder = os.path.join(output_path, path, folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    sparse_folder = os.path.join(output_folder, "sparse")
    if not os.path.exists(sparse_folder):
        os.makedirs(sparse_folder)
    dense_folder = os.path.join(output_folder, "dense")
    if not os.path.exists(dense_folder):
        os.makedirs(dense_folder)
    database_path = os.path.join(output_folder, 'database.db')

    cmd = 'colmap feature_extractor --database_path {0} --image_path {1}'.format(database_path, input_folder)
    print(cmd)
    os.system(cmd)
    cmd = 'colmap exhaustive_matcher --database_path {0}'.format(database_path)
    print(cmd)
    os.system(cmd)
    cmd = 'colmap mapper --database_path {0} --image_path {1} --output_path {2}'.format(database_path, input_folder, sparse_folder)
    print(cmd)
    os.system(cmd)
    cmd = 'colmap image_undistorter --image_path {0} --input_path {1} --output_path {2} --output_type COLMAP'.format(input_folder, os.path.join(sparse_folder, '0'), dense_folder)
    print(cmd)
    os.system(cmd)
    time.sleep(1.0)

