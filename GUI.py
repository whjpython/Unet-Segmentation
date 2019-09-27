"""
利用自带的模块，编写了简单的选择模型，测试文件的GUI界面。
"""

from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=16)
from tkinter import *
from tkinter.filedialog import askdirectory,askopenfilename
from predict import main_p
import threading
import numpy as np
import time
from unet import Unet
import glob
import os


def selectPPath():
    path_ = askopenfilename()
    pic_path.set(path_)

def selectMPath():
    path_ = askdirectory()
    mask_path.set(path_)

root = Tk()
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
ww = 400
wh = 150
x = (sw-ww) / 2
y = (sh-wh) / 2
root.geometry("%dx%d+%d+%d" %(ww,wh,x,y))#居中显示代码
root.title('U-NET')
pic_path = StringVar()#内部定义的字符串变量类型
mask_path = StringVar()
change = StringVar()
Label(root,text = "权重路径:").grid(row = 0, column = 0)#标签
A = Entry(root, textvariable = pic_path)#
A.grid(row = 0, column = 1)
Button(root, text = "权重文件", command = selectPPath,width=10).grid(row = 0, column = 2,padx=5,pady=5)
Label(root,text = "目标路径:").grid(row = 1, column = 0)
B = Entry(root, textvariable = mask_path)
B.grid(row = 1, column = 1)

Button(root, text = "识别文件夹", command = selectMPath,width=10).grid(row = 1, column = 2,padx=5,pady=5)
C = Label(root, textvariable=change,fg = 'red')#textvariable代替text
C.grid(row=2, column=1)

def show():
    modelpath = A.get()
    pic =B.get()
    allpic = glob.glob(os.path.join(pic,'*.tif'))
    lastpic = allpic[-1]
    d,n = os.path.split(lastpic)
    lastpic_save = os.path.join(d,'result'+'/'+n)
    delete_path  = os.path.join(d,'result')
    if os.path.exists(delete_path):
        pp = os.listdir(delete_path)
        for x in pp:
            delete = os.path.join(delete_path,x)
            os.remove(delete)
        os.removedirs(delete_path)
    if pic and modelpath:
        print(pic)
        change.set('执行中...')
        model = Unet((256, 256, 3), 2)
        model.load_weights(modelpath)
        model.predict(np.zeros((2, 256, 256, 3)))#提前预测
        th = threading.Thread(target=main_p,args=(model,pic))
        th.setDaemon(True)
        th.start()
        def xx():
            a = '->'
            while True:
                for x in range(5):
                    a = x * '->'
                    s = '程序运行中'+a
                    change.set(s)
                    time.sleep(0.5)
                    if x == 4:
                        a = '->'
                if os.path.exists(lastpic_save):
                    change.set('识别完成！')
                    os.startfile(delete_path)
                    break
        th1 = threading.Thread(target=xx)
        th1.setDaemon(True)
        th1.start()
    else:
        change.set('文件未选中！')


bb = Button(root, text = "执行", width=10,height=2,command = show,fg='blue').grid(row = 2, column = 0,padx=30,pady=5)

Button(root,text='退出',width=10,height=2,command=root.quit,fg='blue').grid(row=2,column=2,padx = 5,pady=10)
root.mainloop()
