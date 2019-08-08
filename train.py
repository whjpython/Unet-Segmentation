from data import get_train_val,train_data,val_data
from unet import Unet
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import loss
import os


epochs = 20
bitchs = 8
num = 3
input_shape = (256,256,3)
mode1_path = 'model.h5'

def make_callbacks():
    lr_epoch_decay = 10
    init_lr = 1e-2
    best_model_checkpoint = ModelCheckpoint(monitor='val_mean_iou', mode='max',
                                            filepath=mode1_path, save_best_only=True,
                                            save_weights_only=False)  # 保存模型
    callbacks = [best_model_checkpoint]
    if lr_epoch_decay:  # 每个epoch都会降低lr
        def get_lr(epoch):
            w = epoch // 10
            lr = init_lr / (lr_epoch_decay ** w)
            if lr < 1e-10:
                lr = 1e-10
            return lr

        callback = LearningRateScheduler(get_lr)
        callbacks.append(callback)

    return callbacks




model = Unet(input_shape,num)
model.summary()
if os.path.exists(mode1_path):#继续训练
    model.load_weights(mode1_path)
callbacks = make_callbacks()
path = r'F:\cmm\yumi\pic'
train_set,val_set = get_train_val(path,'tif')
train = train_data(train_set,bitchs)
val = val_data(val_set,bitchs)
alltrain = len(train_set)
allval = len(val_set)
loss_f = loss.mean_iou
metrics = [loss_f]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
A = model.fit_generator(generator=train,steps_per_epoch= alltrain//bitchs,epochs=epochs,verbose=1,
                        validation_data=val,validation_steps= allval//bitchs,callbacks=callbacks,max_q_size=1)
