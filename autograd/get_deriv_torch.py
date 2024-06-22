import torch
import numpy as np

def func(x):
    return x[0]**3 + 2*x[1]**2 + 4*x[2] + 5*x[3]

# 示例输入，将numpy数组转换为PyTorch张量
x_input_np = np.array([1.0, 1.0, 1.0, 1.0])
x_input = torch.tensor(x_input_np, dtype=torch.float32, requires_grad=True)

# 计算函数值
output = func(x_input)

# 计算梯度
output.backward()

# 打印结果
print("函数值:", output.item())
print("关于 x 的梯度:", x_input.grad.numpy())

# ------------------------------------------------------------

import torch
import numpy as np

def func(x):
    return (
        x[0]**3 + x[0]*x[1]
        + 2*x[1]**2
        + 4*x[2]
        + 5*x[3]
    )

# 示例输入，将numpy数组转换为PyTorch张量
x_input_np = np.array([1.0, 1.0, 1.0, 1.0])
x_input = torch.tensor(x_input_np, dtype=torch.float32, requires_grad=True)

# 计算函数值
output = func(x_input)

# 计算一阶梯度
output.backward(create_graph=True)

# 打印函数值和一阶梯度
print("函数值:", output.item())
print("关于 x 的一阶梯度:", x_input.grad.detach().numpy())

# 保存一阶梯度
first_order_grad = x_input.grad.clone()

# 计算每个一阶梯度分量对输入变量的二阶梯度
second_order_grads = []

for i in range(len(first_order_grad)):
    # 对每个一阶梯度分量进行反向传播，计算二阶梯度
    first_order_grad[i].backward(retain_graph=True)
    second_order_grads.append(x_input.grad.clone().detach().numpy())
    
    # 清除梯度，为计算下一个分量的梯度做准备
    x_input.grad.zero_()

second_order_grads = np.array(second_order_grads)

# 打印结果
print("关于 x 的一阶梯度:", first_order_grad.detach().numpy())
print("关于 x 的二阶梯度:")
print(second_order_grads)
