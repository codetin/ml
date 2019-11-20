from __future__ import print_function
import torch

# 判断CUDA是否可用,如果不可用,则使用CPU作为运算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 穿件一个5*3的空矩阵
x = torch.empty(5, 3, device=device)
print(x)

# 创建一个5*3的随机小数矩阵
x = torch.rand(5, 3, device=device)
print(x)

# 穿件一个5*3的长整形矩阵,设置dtype
x = torch.zeros(5, 3, dtype=torch.long, device=device)
print(x)

# 设置一个数组,包含5.5与3,默认数值会被小数化
x = torch.tensor([5.5, 3], device=device)
print(x)

# 设置一个全为1的5*3矩阵
x = x.new_ones(5, 3, dtype=torch.double, device=device)
print(x)

# 设置一个浮点矩阵
x = torch.randn_like(x, dtype=torch.float, device=device)   # override dtype!
print(x)
# 打印矩阵大小,矩阵大小为一个数组
print(x.size())

# 设置另一个5*3的矩阵,并将x,7相加
y = torch.rand(5, 3, device=device)
print(y)
print(torch.add(x, y))

# 设置一个空矩阵,将相加结果输出至新矩阵中
result = torch.empty(5, 3, device=device)
torch.add(x, y, out=result)
print(result)
print(result[:, 1])

# 设置一个4*4随机矩阵
x = torch.randn(4, 4, device=device)
# view函数将转化矩阵,只有一个参数的view函数将矩阵转化为一维数组,view的切分不会抛弃数据,比如保证转化后维度的总大小等于转化前的总大小
y = x.view(16)
print(y)
# view函数通过两个参数将矩阵转化成其他形状
z = x.view(-1, 8)  # -1意味着此维度的大小由其他维度的数值反推,例如这里第二维度为8,则4*4维度的转化后成为2*8维度
print(z)
q = x.view(-1, 2)
print(q)
