import skimage.io
import glob
import os
from keras.preprocessing.image import img_to_array
import random
import numpy as np
def num_class(a):
    a = int(a)
    return a

def label_to_one_hot_batch(label_batch):  # 变成one_hot编码
    if label_batch.ndim == 4:  # 数组维度如果是4
        label_batch = np.squeeze(label_batch, axis=3)  #

    nb_labels = 3  # 标签种类
    shape = np.concatenate([label_batch.shape, [nb_labels]])  # 合并
    one_hot_batch = np.zeros(shape)
    for label in range(nb_labels):
        one_hot_batch[..., label][label_batch == label] = 1.
    return one_hot_batch
def get_train_val(path,sign,val_rate = 0.25):#制作验证集和训练集
    train_set = []
    val_set  = []
    pic_data = glob.glob(os.path.join(path, '*.%s' % sign))
    random.shuffle(pic_data)#随机排序训练集
    total_num = len(pic_data)
    val_num = int(val_rate * total_num)#验证集占原始数据的25%
    for i in range(len(pic_data)):
        if i < val_num:#添加到验证集
            val_set.append(pic_data[i])
        else:
            train_set.append(pic_data[i])
    return train_set,val_set
def train_data(train_set,batch_size):
    while True:#必须是无限迭代
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(train_set))):
            batch += 1
            img = skimage.io.imread(train_set[i]).astype(np.float32)
            #img = img_to_array(img)  # 图片转化为三维float32
            train_data.append(img)
            label = skimage.io.imread(train_set[i].replace('pic','mask').replace('tif','png'))
            #label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                train_label = label_to_one_hot_batch(train_label)
                #生成器，每次返回，但是并不执行下面的，下一步才执行
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0
def val_data(val_set,batch_size):
    while True:
        val_data =[]
        val_label =[]

        batch = 0
        for i in (range(len(val_set))):
            batch += 1
            img = skimage.io.imread(val_set[i]).astype(np.float32)
            #img = img_to_array(img)  # 图片转化为三维float32
            val_data.append(img)
            label = skimage.io.imread(val_set[i].replace('pic','mask').replace('tif','png'))
            #label = img_to_array(label)#转换
            val_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                val_data = np.array(val_data)
                val_label = np.array(val_label)
                val_label = label_to_one_hot_batch(val_label)
                #生成器，每次返回，但是并不执行下面的，下一步才执行
                yield (val_data, val_label)
                val_data = []
                val_label = []
                batch = 0
