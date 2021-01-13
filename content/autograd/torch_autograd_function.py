import torch
import torch.nn as nn
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input_):
        # 在forward中定义这个运算的forward计算过程
        # 同时可以保存任何在反向传播中需要用到的值
        ctx.save_for_backward(input_)
        output = input_.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

from torch.autograd import Variable
input_ = Variable(torch.randn(1))

output_ = MyReLU.apply(input_)

# 这个relu对象，就是output_.creator，即这个relu对象将output与input连接起来，形成一个计算图
# print(relu)
print(input_)
print(output_)