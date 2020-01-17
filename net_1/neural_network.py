'''
模块说明：
本模块定义了神经网络的基本属性
'''
import numpy
import scipy.special
import net_1.define

class neuralnetwork:

    #给定参数 ：输入、隐藏、输出的节点数以及学习率
    def __init__(self,inputnodes,hiddennodes,outputnodes,learnrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learnrate = learnrate

        #初始化权重（简单法）
        # self.w_ixh = (numpy.random.rand(self.hiddennodes, self.inputnodes) - 0.5)
        # self.w_hxo = (numpy.random.rand(self.outputnodes,self.hiddennodes) - 0.5)
        #初始化权重（正态分布法）
        self.w_ixh = numpy.random.normal(0.0,pow(self.hiddennodes,-0.5),(self.hiddennodes,self.inputnodes))
        self.w_hxo = numpy.random.normal(0.0,pow(self.outputnodes,-0.5),(self.outputnodes,self.hiddennodes))

        #定义激活函数
        self.activation = lambda x: scipy.special.expit(x)

    #计算函数
    def query(self,inputs_list):
        #转化输入,输入值在 0.01 - 1 之间
        inputs = numpy.array(inputs_list,ndmin=2).T

        #输入层与隐藏层的输出
        hidden_inputs = numpy.dot(self.w_ixh,inputs)
        hidden_outputs = self.activation(hidden_inputs)

        #隐藏层与输出层的输出
        out_inputs = numpy.dot(self.w_hxo,hidden_outputs)
        out_output = self.activation(out_inputs)

        #返回计算结果
        max_out = max(out_output)
        for i in range(len(out_output)):
            if out_output[i] == max_out:
                return net_1.define.out_sit[i]

    def forword_query(self,label_order):
        output = numpy.zeros(len(net_1.define.out_sit)) + 0.01
        output[label_order] = 0.99
        out_in = numpy.array(output).T
        hidden_in =numpy.dot(self.w_hxo.T,out_in)
        in_in = numpy.dot(self.w_ixh.T,hidden_in)
        return in_in

    def train(self,inputs_list,targets_list):
        # 转化输入与真实结果
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        # 输入层与隐藏层的输出
        hidden_inputs = numpy.dot(self.w_ixh, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        # 隐藏层与输出层的输出
        out_inputs = numpy.dot(self.w_hxo, hidden_outputs)
        out_output = self.activation(out_inputs)

        #计算误差并进行前馈误差校正
        out_error = targets - out_output
        hidden_error = numpy.dot(self.w_hxo.T,out_error)

        self.w_hxo += self.learnrate * numpy.dot((out_error * out_output * (1.0 - out_output)),numpy.transpose(hidden_outputs))
        self.w_ixh += self.learnrate * numpy.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))








