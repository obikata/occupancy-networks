import torch
from occupancynet import *

model = OccupancyNet('regnet', embed_dim=64, grid_size=16)

# Left and right images of stereo cameras
left_image = torch.randn(1, 3, 256, 256)
right_image = torch.randn(1, 3, 256, 256)

occupancy_map = model(left_image, right_image)
print(occupancy_map.shape)