import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd

import torch
import numpy as np
from utils.others import *

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def crop(img_root, mask_root, name, save_file):
    img = os.path.join(img_root, name)
    mask = os.path.join(mask_root, name)
    # print('mask name', mask)
    img_integeration = Image.open(img)
    mask_integeration = Image.open(mask)
    mask_integeration = mask_integeration.convert('1')
    img_integeration_numpy = np.array(img_integeration)
    mask_integeration_numpy = np.array(mask_integeration)
    mask = np.expand_dims(mask_integeration_numpy, axis=2)

    merge1 = img_integeration_numpy * (mask > 0)
    merge1 = Image.fromarray(merge1)
    merge1.save(os.path.join(save_file, name))

    return merge1


def main(k, date):
    base_root = '/home/huangjq/PyCharmCode/4_project/1_UNet/B4_attUnetv4/results/images'
    root = os.path.join(base_root, date)
    root = os.path.join(root, 'k' + str(k))

    input_root = os.path.join(root, 'input')
    mask_root = os.path.join(root, 'key_areas_mask')

    save_file = os.path.join(root, 'crop_input')
    # print(save_file)
    mkfile(save_file)

    name_list = os.listdir(input_root)

    for i in range(len(name_list)):
        mask = crop(input_root, mask_root, name_list[i], save_file)
        # plt.imshow(mask)
        # plt.show()
        # break


if __name__ == '__main__':
    date = '0822v1'
    k = 0
    main(k, date)
