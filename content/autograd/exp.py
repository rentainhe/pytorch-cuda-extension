import torch
import torch.nn as nn
from torch.autograd import Function

# 自定义计算 e^x
class Exp(Function):

    @staticmethod
    def forward(ctx, x):
        result = x.exp()
        ctx.save_for_backward(result)
        # 保存所需内容，以备backward时使用
        # 所需的结果会被保存在saved_tensors元组中；此处仅能保存tensor类型变量，若其余类型变量（Int等）
        # 可直接赋予ctx作为成员变量，也可以达到保存效果
        return result

    @staticmethod
    def backward(ctx, grad_output):  # 需要自定义梯度回传
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput 就是输入的参数 grad_output
        # 因此只需求exp的导数，在乘以grad_output
        # exp的导数就是本身
        result, = ctx.saved_tensors  # 取出forward中保存的result
        return grad_output * result
