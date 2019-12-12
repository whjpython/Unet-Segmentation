"""
利用自带的模块，编写了简单的选择模型，测试文件的GUI界面。
"""

from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=16)
from tkinter import *
from tkinter.filedialog import askdirectory,askopenfilename,askopenfilenames
from predict_best import P
from unet import Unet,myunet
import glob
import os

#按钮执行命令函数，获取文件名称
def selectPPath():
    path_ = askopenfilename()
    pic_path.set(path_)
#按钮执行命令函数，获取文件路径
def selectMPath():
    path_ = askopenfilenames()
    mask_path.set(path_)

def selectOPath():
    path_ = askdirectory()
    out_path.set(path_)

root = Tk()
sw = root.winfo_screenwidth()#
sh = root.winfo_screenheight()
ww = 400
wh = 220
x = (sw-ww) / 2
y = (sh-wh) / 2
root.geometry("%dx%d+%d+%d" %(ww,wh,x,y))#居中显示代码
root.title('中科光启深度学习系统')
root.resizable(width=False, height=False)#长宽不能变
pic_path = StringVar()#内部定义的字符串变量类型
mask_path = StringVar()
change = StringVar()
out_path = StringVar()
num_class = IntVar()
sign = BooleanVar()
Label(root,text = "权重路径:").grid(row = 0, column = 0)#标签
A = Entry(root, textvariable = pic_path)#文本框
A.grid(row = 0, column = 1)
Button(root, text = "权重文件", command = selectPPath,width=10).grid(row = 0, column = 2,padx=5,pady=5)
Label(root,text = "目标路径:").grid(row = 1, column = 0)
B = Entry(root, textvariable = mask_path)
B.grid(row = 1, column = 1)

Button(root, text = "识别文件", command = selectMPath,width=10).grid(row = 1, column = 2,padx=5,pady=5)
C = Label(root, textvariable=change,fg = 'red').grid(row=4, column=1)#textvariable代替text

Label(root,text="输出路径").grid(row=2,column=0)
Button(root,text="输出路径",command = selectOPath,width=10).grid(row=2,column = 2,padx=5,pady=5)
F = Entry(root, textvariable = out_path)#文本框
F.grid(row = 2, column = 1)

Label(root,text = "识别种类:").grid(row = 3, column = 0)
D = Spinbox(root,from_=2, to=100, increment=1,textvariable=num_class,width=10).place(x = 140,y=128)
Label(root,text = "数据增强:").place(x = 230 ,y =128)
E = Radiobutton(root,text='是',variable= sign,value = True).place(x = 282,y=128)
Radiobutton(root,text='否',variable= sign,value = False).place(x = 322,y=128)




def get_value():

    modelpath = A.get()
    picinit = B.get()
    outpath = F.get()
    #pic = list(pic)
    pic = picinit.split(' ')
    sign1 = sign.get()
    num_class1 = num_class.get()
    if not modelpath or not pic:
        change.set('请选择文件')

    if modelpath and pic:
        change.set('执行中...')
        root.update()
        #allpic = glob.glob(os.path.join(pic,'*.tif'))
        lastpic = pic[-1]
        model = Unet((256, 256, 3), num_class1)
        #model = myunet((256,256,3),num_class1)
        model.load_weights(modelpath)
        d,n = os.path.split(lastpic)
        lastpic_save = os.path.join(outpath+'/'+n)#最后一个文件
        # delete_path  = os.path.join(d,'result')#保存文件目录
        # if os.path.exists(delete_path):
        #     pp = os.listdir(delete_path)
        #     for x in pp:
        #         delete = os.path.join(delete_path,x)
        #         os.remove(delete)
        #     os.removedirs(delete_path)
        W = P(num_class1)
        W.main_p(model,pic,outpath,changes=sign1)
        if os.path.exists(lastpic_save):
            change.set('识别完成！')
            os.startfile(outpath)


Button(root, text = "执行", width=10,height=2,command = get_value,fg='blue').grid(row = 4, column = 0,padx=30,pady=5)
Button(root,text='退出',width=10,height=2,command=root.quit,fg='blue').grid(row=4,column=2,padx = 5,pady=10)
root.mainloop()
