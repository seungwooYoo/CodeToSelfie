import pandas as pd
import numpy as np
# import ipdb
import os
import shutil


def divide_image_by_class(orig_img_folder, save_folder):
    """
        For analyzing the images, divide images by their classes
    """
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    data = pd.read_csv('./data/train.csv')
    data_id = data.pop('id').values.tolist()
    y = data.pop('species').values.tolist()

    unique_class = np.unique(y).tolist()
    cnt_labels = np.zeros((len(unique_class), 1)).ravel().tolist()
    class_label = zip(unique_class, cnt_labels[:])
    dict_class_cnt = dict(class_label)

    # cur_id : image name
    # save each image into same class folder
    for cur_id, cur_species in zip(data_id, y):
        if cur_species in dict_class_cnt:
            dict_class_cnt[cur_species] += 1
            img_name = str(cur_id) + '.jpg'
            cur_save_folder = os.path.join(save_folder, cur_species)
            if not os.path.exists(cur_save_folder):
                os.mkdir(cur_save_folder)
            cur_save_name = os.path.join(cur_save_folder, img_name)
            orig_file_name = os.path.join(orig_img_folder, img_name)
            if os.path.exists(cur_save_folder) and os.path.exists(orig_file_name):
                shutil.copyfile(orig_file_name, cur_save_name)
        else:
            print 'Failed to find cur specieis\n'

    print dict_class_cnt

if __name__ == '__main__':
    orig_img_folder = './data/images/'
    save_img_folder = './data/classwise/'
    divide_image_by_class(orig_img_folder, save_img_folder)
