import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras.backend as K

def iou(y_pre: np.ndarray, y_true: np.ndarray) -> 'dict':#混淆矩阵，真实类别分布规律，预测类别分布规律，加上总共类别数量
    # cm是混淆矩阵
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pre,
        labels=[0, 1, 2])


    result_iou = [
        cm[i][i] / (sum(cm[i, :]) + sum(cm[:, i]) - cm[i, i]) for i in range(len(cm))#长度就是类别数,算对角线，对角线不为0，旁边为0准确
    ]

    metric_dict = {}
    metric_dict['IOU_其他/other'] = result_iou[0]#数值越接近1越准确
    metric_dict['IOU_植物/plant'] = result_iou[1]
    metric_dict['IOU_道路/road']  = result_iou[2]
    #metric_dict['IOU_建筑/arch']  = result_iou[3]
    #metric_dict['IOU_水体/water'] = result_iou[4]

    metric_dict['iou'] = np.mean(result_iou)#求平均数
    metric_dict['accuracy'] = sum(np.diag(cm)) / sum(np.reshape(cm, -1))#diag二维只求对角线 对角线的元素 / 混淆矩阵所有元素 越准确值越接近与1

    return metric_dict
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):#大于0.5
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 3)#一个是评价分数，一个是返回的真实值和预测值后面是种类
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)