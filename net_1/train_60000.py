'''
模块说明：
本模块是对训练样本60000的循环训练，循环次数参见define模块，建议小于 8 ，过大会导致过拟合
'''
import struct
import numpy
from net_1 import neural_network,define

#创建神经网络的实例
n = neural_network.neuralnetwork(define.in_nodes,define.hidden_nodes,define.out_nodes,define.learn_rate)
train_time = 0

#*******************打开训练集合并得到返回值 images 和 labels *************
def load_mnist():
    #设定路径
    labels_path = 'train-labels.idx1-ubyte'
    images_path = 'train-images.idx3-ubyte'

    #读取labels的数值，60000个
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        print("总共：",n,)
        labels = numpy.fromfile(lbpath,dtype=numpy.uint8)

    #读取imags的数值，60000个长度为748的一维数组
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        print("总共：",num,"\n分为",rows,"x",cols,"列")
        images = numpy.fromfile(imgpath,dtype=numpy.uint8).reshape(len(labels), 784)

    #返回其值
    return images, labels

images,labels = load_mnist()

learn_time = 1

#****************开始训练*****************
for time in range(define.learn_time):
    for i in range(len(labels)):
        print("训练次数：", learn_time)
        learn_time += 1

        inputs = numpy.asfarray(images[i]) / 255 * 0.99 + 0.01

        targets = numpy.zeros(define.out_nodes) + 0.01
        targets[int(labels[i])] = 0.99

        n.train(inputs,targets)
    print("训练完成")

#*************写入权重***************
file = open('whight_ixh.txt','w')
for i in n.w_ixh:
    for j in i[0:len(i)-1]:
        file.write(str(j))
        file.write(',')
    file.write(str(i[len(i)-1]))
    file.write('\n')
print("ixh权重写入完成！")
file.close()

file = open('whight_hxo.txt','w')
for i in n.w_hxo:
    for j in i[0:len(i)-1]:
        file.write(str(j))
        file.write(',')
    file.write(str(i[len(i)-1]))
    file.write('\n')
print("hxo权重写入完成！")
file.close()



