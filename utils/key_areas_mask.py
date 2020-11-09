import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd

import torch
import numpy as np


def Leave_target(a):
    a_tensor = torch.from_numpy(a)
    a2 = torch.squeeze(a_tensor)
    a3 = torch.unsqueeze(a2, 1)
    at = a2.t()
    x_max = torch.max(at[0]) - torch.min(at[0])
    y_max = torch.max(at[1]) - torch.min(at[1])
    object_xy = []
    if x_max > y_max:  # x的最大值大于y的最大值
        tag = 'horizontal'   # 分割图是水平方向的
        y1 = torch.ge(at[1], 470)  # 比较y轴大于470的，返回的是tensor（布尔值）
        y2 = torch.le(at[1], 10)
        # 遍历x
        at_del_same = np.unique(at[0])
        for i in range(at_del_same.shape[0]):
            x_val = at_del_same[i]
            #             print('x_val=', x_val)
            equal_index = (at[0] == x_val).nonzero()  # 找到x相同的索引，比较y值，留下大的
            y = []  # 记下x值相同，各自对应的y值
            for k in range(equal_index.shape[0]):
                index = equal_index[k].cpu().numpy()[0]
                y.append(at[1][index].item())
            if y1.sum() > y2.sum():  # 留下大的y
                y_val = min(y)
                x_val = x_val.item()
                object_xy.append([x_val, y_val])
            else:  # 留下xiao的y
                y_val = max(y)
                x_val = x_val.item()
                object_xy.append([x_val, y_val])
    else:
        tag = 'vertical'
        x1 = torch.ge(at[0], 470)  # 比较y轴的，返回的是tensor（布尔值）
        x2 = torch.le(at[0], 10)

        # 遍历y
        at_del_same = np.unique(at[1])
        for i in range(at_del_same.shape[0]):
            y_val = at_del_same[i]
            equal_index = (at[1] == y_val).nonzero()  # 找到y相同的索引，比较y值
            x = []  # 记下y值相同，各自对应的x值
            for k in range(equal_index.shape[0]):
                index = equal_index[k].cpu().numpy()[0]
                x.append(at[0][index].item())
            if x1.sum() > x2.sum():  # 留下大的y
                x_val = min(x)
                y_val = y_val.item()
                object_xy.append([x_val, y_val])
            else:
                x_val = max(x)
                y_val = y_val.item()
                object_xy.append([x_val, y_val])
    object_xy = np.array(object_xy)
    object_xy = np.expand_dims(object_xy, axis=1)
    return object_xy, tag


"""
def strip(tag, contours):
    a = torch.from_numpy(contours)
    b = torch.full(a.shape, 40, dtype=int)
    c = a - b
    d = a + b
    d = np.flipud(d)
    e = np.concatenate((c, d), axis=0)
    return e

"""


def strip(tag, contours):
    a = np.copy(contours)
    a = np.squeeze(a)
    k = contours.shape[0]
    arr_0 = np.full((1, k), 0).T
    arr_100 = np.full((1, k), 60).T
    # print(contours.shape)
    if tag == 'horizontal':  # 在y方向上加减像素
        a[:, [0]] = arr_0
        a[:, [1]] = arr_100
        # print('horizontal', a.shape, '\n', a)
    else:  # 在x方向上加减像素
        a[:, [0]] = arr_100
        a[:, [1]] = arr_0
        # print('ve', a.shape, '\n', a)
    a = np.expand_dims(a, axis=1)
    c = contours - a
    d = contours + a
    d = np.flipud(d)
    e = np.concatenate((c, d), axis=0)
    return e

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)



def save_key_areas_mask(image_root, gt_root, pred_root, name):
    """
    输入是三个图片的路径
    """
    image_root = os.path.join(image_root, name)
    gt_root = os.path.join(gt_root, name)
    pred_root = os.path.join(pred_root, name)

    im = cv2.imread(image_root)
    gt = cv2.imread(gt_root)
    pr = cv2.imread(pred_root)
    im1 = im.copy()
    gt1 = gt.copy()
    pr1 = pr.copy()
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    gt1_gray = cv2.cvtColor(gt1, cv2.COLOR_BGR2GRAY)
    pr1_gray = cv2.cvtColor(pr1, cv2.COLOR_BGR2GRAY)

    # contours, h = cv2.findContours(gt1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # areas_gt = [cv2.contourArea(each_conts) for each_conts in contours]
    # k_gt = 0
    # for each_conts in areas_gt:
    #     if each_conts > 10000:
    #         contours_gt = contours[k_gt]
    #         contours_gt = Leave_target(contours_gt)
    #         # cv2.drawContours(im1_gray, [contours_gt], 0, (255, 0, 0), 5, lineType=cv2.LINE_8)
    #     k_gt += 1

    contours, h = cv2.findContours(pr1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 删除连通域小于2000的轮廓
    areas = [cv2.contourArea(each_conts) for each_conts in contours]
    k = 0
    for each_conts in areas:
        if each_conts > 9000:
            contours_pre = contours[k]
            contours_p, tag = Leave_target(contours_pre)
            contours_pre = strip(tag, contours_p)
            # cv2.fillPoly(im1_gray, [contours_pre], color=(0, 0, 0))
            cv2.drawContours(im1_gray, [contours_p], 0, (0, 255, 0), 5, lineType=cv2.LINE_8)
            #             cv2.drawContours(pr1, [contours_pre], 0, (0, 255, 0), 5, lineType=cv2.LINE_8)
            # cv2.circle(pr1, (800, 800), radius=50, color=(255, 0, 0),
            #            thickness=-50)
        k += 1

    im = Image.open(image_root)
    mask = draw(contours_pre, im.size[0], im.size[1])

    im1_gray = Image.fromarray(im1_gray)
    return im1_gray, mask


def draw(contours, width, height):
    img = np.zeros((height, width, 3), np.uint8)

    cv2.fillPoly(img, [contours], (255, 255, 255))

    # plt.imshow(img)
    # plt.axis('off')
    img = Image.fromarray(img)
    # img.save('draw.JPG')
    # plt.show()
    return img


if __name__ == '__main__':

    date = '0822v1'
    k = 0
    base_root = '/home/huangjq/PyCharmCode/4_project/1_UNet/B4_attUnetv4/results/images'
    root = os.path.join(base_root, date)
    root = os.path.join(root, 'k' + str(k))
    input_root = os.path.join(root, 'input')
    label_root = os.path.join(root, 'label')
    ouput_root = os.path.join(root, 'output')

    save_contours = os.path.join(root, 'contours')
    save_mask = os.path.join(root, 'key_areas_mask')

    mkfile(save_contours)
    mkfile(save_mask)

    name_list = os.listdir(input_root)

    for i in range(len(name_list)):
        contours, mask = save_key_areas_mask(input_root, label_root, ouput_root, name_list[i])

        plt.imshow(contours)
        plt.axis('off')
        plt.imshow(mask)
        plt.axis('off')

        # contours.save(os.path.join(save_contours, name_list[i]))
        # mask.save(os.path.join(save_mask, name_list[i]))
        plt.show()
        break
