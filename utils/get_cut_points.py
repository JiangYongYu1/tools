import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 定义Mish函数
def mish(x):
    return x * torch.tanh(F.softplus(x))

def elu(x):
    return torch.nn.functional.elu(x)

def silu(x):
    return torch.nn.functional.silu(x)

def gelu(x):
    return torch.nn.functional.gelu(x)

class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()


    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.gelu(input, approximate="tanh")

def pytorchgelutanh(x):
    return PytorchGELUTanh()(x)


# 计算曲率的函数
def get_curvature(x):
    x.requires_grad_(True)
    
    # 计算函数值
    y = silu(x)
    
    # 计算一阶导数
    first_derivative = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    
    # 计算二阶导数
    second_derivative = torch.autograd.grad(first_derivative, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    
    # 计算曲率
    curvature = torch.abs(second_derivative) / (1 + first_derivative**2)**(3/2)
    
    return curvature

# 创建x值
x = torch.linspace(-50, 50, 100000)

# 计算曲率
curvature = get_curvature(x)

diff = curvature[1:] - curvature[:-1]
# 找到第一个显著变化点和最后一个显著变化点
# 设置阈值来检测显著变化
threshold = 1e-6
significant_changes = torch.where(torch.abs(diff) > threshold)[0]
if len(significant_changes) > 0:
    first_change = x[significant_changes[0]]
    last_change = x[significant_changes[-1]]
    print(f"曲率开始变化的点（从0开始变化）: {first_change.item():.4f}")
    print(f"曲率最后变化的点（变回0）: {last_change.item():.4f}")

# 找出曲率为0的点（即二阶导数为0的点）
zero_points = x[torch.where(curvature < 1e-3)[0]]

# 绘制曲率图
plt.figure(figsize=(12, 8))
plt.plot(x.detach().numpy(), curvature.detach().numpy())
plt.grid(True)
plt.title('Mish函数的曲率')
plt.xlabel('x')
plt.ylabel('曲率')
plt.savefig('mish_curvature.png')

print("曲率接近0的点：", zero_points.detach().numpy())