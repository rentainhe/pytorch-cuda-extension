import torch
# clone

# 1. clone后的返回值是一个中间 Variable, 支持梯度回溯
a = torch.tensor(1.0, requires_grad=True, device="cpu", dtype=torch.float64)
a_ = a.clone()
print(a_)  # grad_fn=<CloneBackward>

# 2. clone后的梯度回传情况
a = torch.tensor(1.0, requires_grad=True)
y = a ** 2
a_ = a.clone()
a_.retain_grad()
z = a_ * 3
y.backward()
z.backward()

print(a_.grad)
print(a.grad)
a.grad.zero_()
print(a.grad)


