from PIL import Image
import matplotlib.pyplot
from net_1 import neural_network,define
import numpy

#创建神经网络的一个实例
n = neural_network.neuralnetwork(define.in_nodes,define.hidden_nodes,define.out_nodes,define.learn_rate)

#获取（最大值的序号）的函数
def get_max(out_list):
    max_num = max(out_list)
    for i in range(0,len(out_list)):
        if out_list[i] == max_num:
            return i

#将输入的值从0-255转化为0.01-1
def get_data_in(data):
    for i in range(len(data)):
            data[i]= data[i]/255*0.99+0.01
    return data

#***************读取权重给神经网络***************
#输入层到隐藏层的权重
file = open('whight_ixh.txt')
whight_ixh = file.readlines()
for i in range(0,len(whight_ixh)-1):
    row = whight_ixh[i].split(',')
    n.w_ixh[i] = row

#隐藏层到输出层的函数
file = open('whight_hxo.txt')
whight_hxo = file.readlines()
for i in range(0,len(whight_hxo)-1):
    row = whight_hxo[i].split(',')
    n.w_hxo[i] = row

#图片的二进制化文本路径
t_path = "the_image\\image.txt"
#图片的路径
p_path = ""

#****************通过图片路径将其转化为0-255的文本******************
def get_txt():
    #输入图片路径，若未输入，则默认为  （ "the_image\\image.png" ）
    while(1):
        try:
            p_path = input("请输入图片路径：")
            break
        except TypeError:
            p_path = input("输入错误，请重新输入图片路径：")

    if p_path == "":
        p_path = "the_image\\image.png"

    #将图片缩小至  in_width * in_hight（此处为28*28）
    #-----！！！！图片要求为in_width * in_hight的整数倍！！！！！
    image = Image.open(p_path)
    image.thumbnail((define.in_width,define.in_hight))

    #以下为转化过程
    file = open(t_path, 'w')

    for i in range(define.in_hight):
        for j in range(define.in_width - 1):
            color = image.getpixel((j,i))
            v = 255 - int((color[0] + color[1] + color[2]) / 3)
            file.write(str(v))
            file.write(',')
        color = image.getpixel((i, define.in_width - 1))
        v = 255 - int((color[0] + color[1] + color[2]) / 3)
        file.write(str(v))
        file.write('\n')
    file.close()
    image.close()

#*****************通过文本将图片显示，并判断其值**************
def get_num():
    #读取文本数据并转化为  int  型数据
    file = open(t_path)
    datas = file.readlines()
    v = []
    for i in range(len(datas)):
        for j in datas[i].split(','):
            v.append(int(j))

    define.show_picture(v)

    #判断数值
    out = n.query(get_data_in(v))
    print('数字是------------》',out)

def forward(num):
    out = n.forword_query(num)
    print("数字为：",num)
    define.show_picture(out)

if __name__ == '__main__':
    for i in define.out_sit:
        forward(i)