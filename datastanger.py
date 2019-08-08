import cv2
import numpy as np
import random

import glob
from tqdm import tqdm
import pandas as pd
import os

size = 256



# 随机窗口采样
def generate_train_dataset(mask_path,sign,image_num = 99999,
                           train_image_path='dataset/train/images/',
                           train_label_path='dataset/train/labels/'):
    '''
    该函数用来生成训练集，切图方法为随机切图采样
    :param image_num: 生成样本的个数
    :param train_image_path: 切图保存样本的地址
    :param train_label_path: 切图保存标签的地址
    :return:
    '''
    images_path = []
    labels_path = []
    # 用来记录所有的子图的数目
    g_count = 1
    allmask = glob.glob(os.path.join(mask_path,"*%s"%sign))
    for one in allmask:
        labels_path.append(one)
        one_pic = one.replace('mask','pic')
        images_path.append(one_pic)

    # 每张图片生成子图的个数
    image_each = image_num // len(images_path)
    image_path, label_path = [], []
    for i in tqdm(range(len(images_path))):
        count = 0
        image = cv2.imread(images_path[i])#读取pic
        label = cv2.imread(labels_path[i], cv2.CAP_MODE_GRAY)#读取mask,灰度图方式
        X_height, X_width = image.shape[0], image.shape[1]#获取图片长
        while count < image_each:#计数
            random_width = random.randint(0, X_width - size - 1)#随机生成宽
            random_height = random.randint(0, X_height - size - 1)#随机生成长
            image_ogi = image[random_height: random_height + size, random_width: random_width + size,:]#裁剪
            label_ogi = label[random_height: random_height + size, random_width: random_width + size]#裁剪

            image_d, label_d = data_augment(image_ogi, label_ogi)#数据增强，同时对标签和图片

            image_path.append(train_image_path+'%05d.png' % g_count)
            label_path.append(train_label_path+'%05d.png' % g_count)
            cv2.imwrite((train_image_path+'%05d.png' % g_count), image_d)#保存图像
            cv2.imwrite((train_label_path+'%05d.png' % g_count), label_d)

            count += 1
            g_count += 1
    df = pd.DataFrame({'image':image_path, 'label':label_path})#数据表格
    df.to_csv('dataset/path_list.csv', index=False)#保存pdf

# 以下函数都是一些数据增强的函数
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(size)]

    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)

    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)

    gamma = np.exp(alpha)

    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((size /2, size / 2), angle, 1)

    xb = cv2.warpAffine(xb, M_rotate, (size, size))

    yb = cv2.warpAffine(yb, M_rotate, (size, size))

    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3))

    return img


def add_noise(img):
    for i in range(size):  # 添加点噪声

        temp_x = np.random.randint(0, img.shape[0])

        temp_y = np.random.randint(0, img.shape[1])

        img[temp_x][temp_y] = 255

    return img


def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)

    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)

    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转

        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    # 双边过滤
    if np.random.random() < 0.25:
        xb =cv2.bilateralFilter(xb,9,75,75)

    #  高斯滤波
    if np.random.random() < 0.25:
        xb = cv2.GaussianBlur(xb,(5,5),1.5)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb

if __name__ == '__main__':
    if not os.path.exists('dataset/train/images'): os.mkdir('dataset/train/images')
    if not os.path.exists('dataset/train/labels'): os.mkdir('dataset/train/labels')
    mask = r'G:\BaiduNetdiskDownload\origin\mask'
    sign = '.png'
    generate_train_dataset(mask,sign)
