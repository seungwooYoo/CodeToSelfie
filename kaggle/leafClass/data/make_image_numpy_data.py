import os
import numpy as np
import cv2
import scipy.misc as smisc


class ImageVariation(object):
    """ Image variation for training
    """
    def __init__(self):
        self.img_size = (64, 64)
        self.flip = True
        self.rotation = np.linspace(-5, 5, 20)

    def VariationData(self, img, numb_variations):
        size = img.shape

        variationed_img = []
        for i in range(numb_variations):
            if self.flip:
                if np.random.rand(1) > 0.5:
                    # do vertical flip
                    new_img = cv2.flip(img, flipCode=1)
                else:
                    new_img = np.copy(img)
            if len(self.rotation) != 0:
                rand_rot_index = np.random.randint(len(self.rotation))
                new_img = smisc.imrotate(new_img, self.rotation[rand_rot_index])
            if self.img_size[0] != size[0] and self.img_size[1] != size[1]:
                new_img = smisc.imresize(new_img, self.img_size, interp='nearest')
                cv2.imshow('orig', img)
                cv2.imshow('test', new_img)
                cv2.waitKey(0)
            variationed_img.append(new_img)
        return variationed_img

def retrieve_all_images(root_path):
    all_img_paths = []
    if root_path == []:
        return []
    else:
        all_img_lists = os.listdir(root_path)
        for path in all_img_lists:
            full_path = os.path.join(root_path, path)
            if os.path.isdir(full_path):
                all_img_paths.extend(retrieve_all_images(full_path))
            elif path.endswith('.png') or path.endswith('.jpg'):
                all_img_paths.append(full_path)
    return all_img_paths

if __name__ == "__main__":
    image_variator = ImageVariation()
    root_img_path = './classwise'
    all_img_paths = retrieve_all_images(root_img_path)

    for img_name in all_img_paths:
        cur_img = cv2.imread(img_name)
        images = image_variator.VariationData(cur_img, 5)
