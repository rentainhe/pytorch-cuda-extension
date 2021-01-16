import torch
# without .detach()
x = torch.tensor([2.], requires_grad=True)

a = torch.add(x, 1)
b = torch.add(x, 2)
y = torch.mul(a, b)

y.backward()

print("without detach")
print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print("grad: ", x.grad, a.grad, b.grad, y.grad)
print()


# with .detach()
x = torch.tensor([2.], requires_grad=True)

a = torch.add(x, 1).detach()
b = torch.add(x, 2)
y = torch.mul(a, b)

y.backward()

print("with detach")
print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print("grad: ", x.grad, a.grad, b.grad, y.grad)
print()