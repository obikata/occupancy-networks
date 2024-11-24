import torch
from occupancynet.model import OccupancyNet

model = OccupancyNet('regnet')

# Left and right images of stereo cameras
left_image = torch.randn(1, 3, 224, 224)
right_image = torch.randn(1, 3, 224, 224)

occupancy_map = model(left_image, right_image)
print(occupancy_map.shape)