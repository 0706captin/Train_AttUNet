import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import pandas as pd


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def save_hd(hd, name, date, k):
    data = {'name': name, 'hd': hd}
    dataframe = pd.DataFrame(data, columns=['name', 'hd'])  # columns自定义列的索引值
    csv_file = os.path.join('./results/csv_file', date)
    csv_file = os.path.join(csv_file, 'k' + str(k))
    mkfile(csv_file)
    csv_name = 'score_' + date + '.csv'
    a = os.path.join(csv_file, csv_name)
    dataframe.to_csv(a)


def Hausdorff_distance(i, image_root, gt_root, pred_root, name, date):
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

    contours, h = cv2.findContours(gt1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas_gt = [cv2.contourArea(each_conts) for each_conts in contours]
    k_gt = 0
    for each_conts in areas_gt:
        if each_conts > 50000:
            contours_gt = contours[k_gt]
            cv2.drawContours(im1_gray, [contours_gt], 0, (255, 0, 0), 3, lineType=cv2.LINE_8)
        k_gt += 1

    contours, h = cv2.findContours(pr1_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 删除连通域小于2000的轮廓
    areas = [cv2.contourArea(each_conts) for each_conts in contours]
    k = 0
    for each_conts in areas:
        if each_conts > 50000:
            contours_pre = contours[k]
            cv2.drawContours(im1_gray, [contours_pre], 0, (0, 255, 0), 3, lineType=cv2.LINE_8)
        k += 1

    plt.imshow(im1_gray)
    plt.axis('off')
    # save 线围成一个圈
    a = Image.fromarray(im1_gray)
    save_dir = './results/contours'
    save_dir = os.path.join(save_dir, date)
    save_dir = os.path.join(save_dir, 'k' + str(i))

    mkfile(save_dir)
    a.save(os.path.join(save_dir, name))
    # plt.show()

    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    try:
        hd = hausdorff_sd.computeDistance(contours_gt, contours_pre)
    except:
        hd = 'None'
    return hd

#
# if __name__ == '__main__':
#
#     root = '../results/images/'
#     date = '0821v1'
#     root = os.path.join(root, date)
#
#     input_root = os.path.join(root, 'input')
#     label_root = os.path.join(root, 'label')
#     ouput_root = os.path.join(root, 'output')
#
#     name_list = os.listdir(input_root)
#
#     # 单张图片的测试
#     # savecontours(input_root, label_root, ouput_root, '002-RI1.JPG')
#     # 批量检查
#     # for i in range(len(name_list)):
#     #     savecontours(input_root, label_root, ouput_root, name_list[i])
#     # break
#
#     hd_list = []
#     for i in range(len(name_list)):
#         hd = Hausdorff_distance(input_root, label_root, ouput_root, name_list[i])
#         hd_list.append(hd)
#         # break
#     save_hd(hd_list, name_list)
#     print('hd_list = ', hd_list)


def save_hd_contours(k, date):
    root = './results/images'
    root = os.path.join(root, date)
    root = os.path.join(root, 'k' + str(k))

    input_root = os.path.join(root, 'input')
    label_root = os.path.join(root, 'label')
    ouput_root = os.path.join(root, 'output')

    name_list = os.listdir(input_root)

    hd_list = []

    for i in range(len(name_list)):
        hd = Hausdorff_distance(k, input_root, label_root, ouput_root, name_list[i], date)
        hd_list.append(hd)
    return hd_list

    # for i in range(len(name_list)):
    #     hd = Hausdorff_distance(k, input_root, label_root, ouput_root, name_list[i], date)
    #     hd_list.append(hd)
    # # save_hd(hd_list, name_list)
    # return hd_list, name_list


if __name__ == '__main__':
    hd_list, name = save_hd_contours(k=0, date='0822v1')
    save_hd(hd_list, name, 'date', 2)
