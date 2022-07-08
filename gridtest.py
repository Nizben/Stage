import torch

input = torch.arange(4*4).view(1, 1, 4, 4).float()
print(input)

d = torch.linspace(-1, 1, 8)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0) # add batch dim

output = torch.nn.functional.grid_sample(input, grid)
print(output) 