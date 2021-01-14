## Extending `torch.autograd`
### 1. torch.autograd.Function
需要重写两个方法, `forward` 和 `backward`
```python
from torch.autograd import Function

class new_operation(Function):
    @staticmethod
    def forward(ctx, x):
        ...
    
    @staticmethod
    def backward(ctx, grad_output):
        ...
```
输入输出解读:
- `ctx`: 必选参数, 可以使用pytorch的autograd engine中定义的一些方法
  - `ctx.save_for_backward()`: 可以用于保存`forward`中的`输入`和`输出`, 用于`backward`
- `grad_output`: 根据`dloss / dx = (dloss / doutput) * (doutput / dx)`, `grad_output`表示`(dloss / doutput)`, 所以在`backward`中只需要计算`doutput/dx`

|example|code|
|:---:|:---:|
| __relu__ |[exp.py]()|
| __exp__ |[relu.py]()|


## Implemented Article
- [Pytorch Docs](https://pytorch.org/docs/master/notes/extending.html?highlight=ctx)
- [Pytorch源码解读 - torch.autograd](https://www.zhihu.com/search?type=content&q=torch.autograd)