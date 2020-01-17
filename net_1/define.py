'''
模块说明：
本模块用于定义该神经网络的一些基本参数
如下
'''
import matplotlib.pyplot
import numpy

#输入节点数
in_nodes = 784

#输入矩阵的宽与高，相乘应等于输入节点数
in_width = 28
in_hight = 28

#输出节点数
out_nodes = 10
#隐藏节点数
hidden_nodes = 500
#学习率
learn_rate = 0.2
#同一样本循环学习次数
learn_time = 3

#输出集合，需要大于等于输出节点数
out_sit = [0,1,2,3,4,5,6,7,8,9]

# 将图片显示出来
def show_picture(data):
    im_array = numpy.asfarray(data).reshape((in_width, in_hight))
    matplotlib.pyplot.imshow(im_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()