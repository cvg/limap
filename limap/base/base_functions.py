import cv2

def imread_camview(camview, set_gray=False):
    imname = camview.image_name()
    if imname == "none":
        raise ValueError("Error! The image name has not been set!")
    img = cv2.imread(imname)
    img = cv2.resize(img, (camview.w(), camview.h()))
    if set_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img



