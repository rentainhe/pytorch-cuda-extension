import torch
import torch.nn as nn
from torch.autograd import Function

class ReLU(Function):

    def forward(ctx, x):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(x)         # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0)               # relu就是截断负数，让所有负数等于0
        return output

    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput 就是输入的参数 grad_output
        # 因此只需求relu的导数，在乘以grad_output
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0                # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        return grad_input

from torch.autograd import Variable
input_ = Variable(torch.randn(1))

output_ = ReLU.apply(input_)

# 这个relu对象，就是output_.creator，即这个relu对象将output与input连接起来，形成一个计算图
# print(relu)
print(input_)
print(output_)