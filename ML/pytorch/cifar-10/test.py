from __future__ import print_function
import torch
import argparse
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', help='Disable CUDA')
parser.add_argument('--site',  help='Disable abvc')
args = parser.parse_args()

print(args.site)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
x = torch.empty(5, 3, device=device)
print(x)
x = torch.rand(5, 3, device=device)
print(x)
x = torch.zeros(5, 3, dtype=torch.long, device=device)
print(x)
x = torch.tensor([5.5, 3], device=device)
print(x)
# new_* methods take in sizes
x = x.new_ones(5, 3, dtype=torch.double, device=device)
print(x)

x = torch.randn_like(x, dtype=torch.float, device=device)   # override dtype!
print(x)
print(x.size())
# result has the same size
y = torch.rand(5, 3, device=device)
print(y)
print(torch.add(x, y))
result = torch.empty(5, 3, device=device)
torch.add(x, y, out=result)
print(result)
print(result[:, 1])

x = torch.randn(4, 4, device=device)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x)
