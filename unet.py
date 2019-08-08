from keras.layers import (
    Input, concatenate, Conv2D, MaxPooling2D,
    BatchNormalization, Activation, UpSampling2D,Dropout,add)
from keras.models import Model

def Unet(input=(256,256,3),numclass = 1):
    inputs = Input(shape=input)

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)  # 3*3的卷积层感受野是（1-1）*1+3=3

    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)  # 激活函数降采样

    # 输出(samples, 128, 128, 64)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)  # （3-1）*1+3=5
    down1 = BatchNormalization()(down1)  # Batch Norm的思路是调整各层的激活值分布使其拥有适当广度
    down1 = Activation('relu')(down1)

    # 输出(samples, 64, 64, 64)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)  # 2*2最大池化层（5-1）*2+2=10
    # 64

    # 输出(samples, 64, 64, 128)
    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)  # （11-1）*1+3=13
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)

    # 输出(samples, 64, 64, 128)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)  # （13-1）*1+3=15
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)

    # 输出(samples, 32, 32, 128)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)  # （15-1）*2+2=30
    # 32

    # 输出(samples, 32, 32, 256)
    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)  # （30-1）*1+2=32
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)  # 34
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    # 输出(samples, 16, 16, 256)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)  # （34-1）*2+2=68
    # 16

    # 输出(samples, 16, 16, 512)
    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)  # 70
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)  # 72
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    # 输出(samples, 8, 8, 512)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)  # （72-1）*2+2=144
    # 8

    # 输出(samples, 8, 8, 1024)
    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)  # 146
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)  # 148
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    # 输出(samples, 16, 16, 1024)
    # at = Attention_layer()(center)
    up4 = UpSampling2D((2, 2))(center)
    # 输出(samples, 16, 16, 1024+512)
    up4 = concatenate([down4, up4], axis=3)
    # 输出(samples, 16, 16, 512)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # at = Attention_layer()(up4)
    # 16

    # 输出(samples, 32, 32, 256)
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    # 输出(samples, 64, 64, 128)
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)

    classify = Conv2D(numclass, (1, 1), activation='softmax')(up1)  # 1*1的卷积层
    model = Model(inputs=inputs, outputs=classify)
    return model

