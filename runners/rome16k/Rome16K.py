import os

import numpy as np


class Rome16K:
    def __init__(self, list_file, component_folder):
        self.image_list = self.load_image_list(list_file)
        self.components = {}
        self.component_names = []
        self.component_ids = (np.ones(len(self.image_list)) * -1).tolist()
        self.load_components(component_folder)

    def load_image_list(self, list_file):
        print(f"Loading bundler list file {list_file}...")
        with open(list_file) as f:
            lines = f.readlines()
        image_names = [line.strip("\n").split(" ")[0] for line in lines]
        return image_names

    def load_component_file(self, component_file):
        with open(component_file) as f:
            lines = f.readlines()
        imname_list = []
        for line in lines:
            imname = line.strip("\n").split(" ")[0]
            imname = os.path.basename(imname)
            imname_list.append(imname)
        return imname_list

    def get_fullname(self, imname):
        imname_db = os.path.join("db", imname)
        if imname_db in self.image_list:
            return imname_db
        imname_query = os.path.join("query", imname)
        if imname_query in self.image_list:
            return imname_query
        else:
            return None

    def load_components(self, component_folder):
        # read from each component file
        flist = os.listdir(component_folder)
        for fname in flist:
            if fname[-4:] != ".txt":
                continue
            self.components[fname] = []
            self.components[fname] = self.load_component_file(
                os.path.join(component_folder, fname)
            )
        self.component_names = list(self.components.keys())

        # map
        for c_id, cname in enumerate(self.component_names):
            for imname in self.components[cname]:
                imname_full = self.get_fullname(imname)
                index = self.image_list.index(imname_full)
                self.component_ids[index] = c_id

    def get_imname(self, img_id):
        return self.image_list[img_id]

    def count_components(self):
        return len(self.component_names)

    def count_images_in_component(self, c_id):
        if isinstance(c_id, str):
            return len(self.components[c_id])
        else:
            return self.count_images_in_component(self.component_names[c_id])

    def get_images_in_component(self, c_id):
        if isinstance(c_id, str):
            images = self.components[c_id]
            images = [self.get_fullname(imname) for imname in images]
            imname_list = []
            for image in images:
                if image is not None:
                    imname_list.append(image)
            return imname_list
        else:
            return self.get_images_in_component(self.component_names[c_id])

    def get_component_id_for_image_id(self, img_id):
        return self.component_ids[img_id]

    def get_component_id_for_image_id_list(self, image_id_list):
        c_ids = [self.component_ids[img_id] for img_id in image_id_list]
        c_id = max(set(c_ids), key=c_ids.count)
        return c_id
