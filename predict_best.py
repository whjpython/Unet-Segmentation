import skimage.io
import numpy as np
import os
from osgeo.gdalconst import *
from osgeo import gdal
import tqdm
import time
from unet import Unet
import glob

back = [0,0,0]
stalk = [0 ,92,230]
twig = [0,255,0]
grain = [128,128,0]
COLOR_DICT = np.array([back,stalk, twig, grain])#上色代码只有4类

#num_class = 4#网络识别类别
class P():
    def __init__(self,number):
        self.number = number

    def predict_x(self,batch_x, model):
        """
        预测一个batch的数据
        """
        batch_y = model.predict(batch_x)#官方预测代码
        return batch_y#返回的是批次的预测(批次，图片长宽，种类）

    def stretch(self,img):#%2線性拉伸
        n = img.shape[2]
        for i in range(n):
            c1 = img[:, :, i]
            c = np.percentile(c1[c1>0], 2)  # 只拉伸大于零的值
            d = np.percentile(c1[c1>0], 98)
            t = (img[:, :, i] - c) / (d - c)
            t *= 65535
            t[t < 0] = 0
            t[t > 65535] = 65535
            img[:, :, i] = t
        return img

    def CreatTf(self,file_path_img,data,outpath):#原始文件，识别后的文件数组形式，新保存文件
        d,n = os.path.split(file_path_img)
        dataset = gdal.Open(file_path_img, GA_ReadOnly)#打开图片只读
        #data = gdal.Open(os.path.join(outpath,'gyey'+n))#打开标签图片
        #data_label = data.ReadAsArray(0, 0, data.RasterXSize, data.RasterYSize)#获取数据
        projinfo = dataset.GetProjection()#获取坐标系
        geotransform = dataset.GetGeoTransform()
        #band = dataset.RasterCount()
        format = "GTiff"
        driver = gdal.GetDriverByName(format)#数据格式
        name = n[:-4]+'_result'+'.tif'#输出文件名字
        dst_ds = driver.Create(os.path.join(outpath,name), dataset.RasterXSize, dataset.RasterYSize,
                                  1, gdal.GDT_Byte )#创建一个新的文件
        dst_ds.SetGeoTransform(geotransform)#投影
        dst_ds.SetProjection(projinfo)#坐标
        dst_ds.GetRasterBand(1).WriteArray(data)
        dst_ds.FlushCache()


    def make_prediction_img(self,x, target_size, batch_size, predict):  # 函数当做变量
        """
        滑动窗口预测图像。

        每次取target_size大小的图像预测，但只取中间的1/4，这样预测可以避免产生接缝。
        """
        # target window是正方形，target_size是边长
        quarter_target_size = target_size // 4
        half_target_size = target_size // 2

        pad_width = (
            (quarter_target_size, target_size),  # 32,128
            (quarter_target_size, target_size),  # 32,128
            (0, 0))

        # 只在前两维pad
        pad_x = np.pad(x, pad_width, 'constant', constant_values=0)  # 填充(x.shape[0]+160,x.shape[1]+160)
        pad_y = np.zeros(
            (pad_x.shape[0], pad_x.shape[1],self.number),
            dtype=np.float32)  # 32位浮点型

        def update_prediction_center(one_batch):
            """根据预测结果更新原图中的一个小窗口，只取预测结果正中间的1/4的区域"""
            wins = []  # 窗口
            for row_begin, row_end, col_begin, col_end in one_batch:
                win = pad_x[row_begin:row_end, col_begin:col_end, :]  # 每次裁剪数组这里引入数据
                win = np.expand_dims(win, 0)  # 喂入数据的维度确定了喂入的数据要求是(n, 256,256,3)
                wins.append(win)
            x_window = np.concatenate(wins, 0)  # 一个批次的数据
            y_window = predict(x_window)  # 预测一个窗格，返回结果需要一个一个批次的取出来
            for k in range(len(wins)):  # 获取窗口编号
                row_begin, row_end, col_begin, col_end = one_batch[k]  # 取出来一个索引
                pred = y_window[k, ...]  # 裁剪出来一个数组，取出来一个批次数据
                y_window_center = pred[
                                  quarter_target_size:target_size - quarter_target_size,
                                  quarter_target_size:target_size - quarter_target_size,
                                  :]  # 只取预测结果中间区域减去边界32[32:96,32:96]

                pad_y[
                row_begin + quarter_target_size:row_end - quarter_target_size,
                col_begin + quarter_target_size:col_end - quarter_target_size,
                :] = y_window_center  # 把预测的结果放到建立的空矩阵中[32:96，32:96]

        # 每次移动半个窗格
        batchs = []
        batch = []
        for row_begin in range(0, pad_x.shape[0], half_target_size):  # 行中每次移动半个[0,x+160,64]
            for col_begin in range(0, pad_x.shape[1], half_target_size):  # 列中每次移动半个[0,x+160,64]
                row_end = row_begin + target_size  # 0+128
                col_end = col_begin + target_size  # 0+128
                if row_end <= pad_x.shape[0] and col_end <= pad_x.shape[1]:  # 范围不能超出图像的shape
                    batch.append((row_begin, row_end, col_begin, col_end))  # 取出来一部分列表[0,128,0,128]
                    if len(batch) == batch_size:  # 够一个批次的数据
                        batchs.append(batch)
                        batch = []
        if len(batch) > 0:
            batchs.append(batch)
            batch = []
        for bat in tqdm.tqdm(batchs, desc='Batch pred'):  # 添加一个批次的数据
            update_prediction_center(bat)  # bat只是一个裁剪边界坐标
        y = pad_y[quarter_target_size:quarter_target_size + x.shape[0],
            quarter_target_size:quarter_target_size + x.shape[1],
            :]  # 收缩切割为原来的尺寸
        return y  # 原图像的预测结果

    def main_p(self,model,allpath,outpath,changes=False):#模型，所有图片路径，输出图片路径
        print('执行预测...')
        #img_p = glob.glob(os.path.join(allpath, "*.%s"%sign))
        for one_path in allpath:
            pic = skimage.io.imread(one_path)
            if changes:
                pic = self.stretch(pic)
            pic = pic.astype(np.float32)
            #pic = img_to_array(pic)
            y_probs = self.make_prediction_img(
                pic, 256, 8,
                lambda xx: self.predict_x(xx, model))  # 数据，目标大小，批次大小，返回每次识别的
            y_preds = np.argmax(y_probs, axis=2)
            d, n = os.path.split(one_path)
            t0 = time.time()
            change = y_preds.astype(np.uint8)
            #outpath = os.path.join(d, 'result')
            # if not os.path.exists(outpath):
            #     os.makedirs(outpath)
            self.CreatTf(one_path, change,outpath)  # 添加坐标系
            img_out = np.zeros(change.shape + (3,))
            for i in range(self.number):
                img_out[change == i, :] = COLOR_DICT[i]#对应上色
            change = img_out / 255
            save_file=os.path.join(outpath,n[:-4]+'_color'+'.png')
            skimage.io.imsave(save_file, change)
            #os.startfile(outpath)
            print('预测耗费时间: %0.2f(min).' % ((time.time() - t0) / 60))
if __name__ == '__main__':
    #model = Unet((256,256,3),num)
    # p = r'YMDD.h5'  # 说明权重所在位置
    # print("网络参数来自: '%s'." % p)
    # model.load_weights(p)
    path = r'G:\19YM\pic'
    # main_p(model,path,changes=True)
