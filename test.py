import torch
from occupancynet.model import OccupancyNet

model = OccupancyNet('regnet', embed_dim=64, grid_size=24)

# Left and right images of stereo cameras
left_image = torch.randn(1, 3, 512, 512)
right_image = torch.randn(1, 3, 512, 512)

occupancy_map = model(left_image, right_image)
print(occupancy_map.shape)