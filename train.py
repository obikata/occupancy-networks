import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from occupancynet import *

class StereoDataset(Dataset):
    def __init__(self, image_pairs, output_tensors):
        self.image_pairs = image_pairs
        self.output_tensors = output_tensors

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        left_image, right_image = self.image_pairs[idx]
        output_tensor = self.output_tensors[idx]
        return (left_image, right_image), output_tensor

def dice_loss(pred, target):
    smooth = 1.
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def loss_function(pred, target):
    # return F.binary_cross_entropy(pred, target)
    return F.binary_cross_entropy(pred, target) + dice_loss(pred, target)  # Combining BCE with Dice for better segmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Left and right images of stereo cameras, and 3D occupancy maps
# TODO: Use simulated stereo camera images and corresponding occupancy maps
left_image = torch.randn(3, 256, 256)
right_image = torch.randn(3, 256, 256)
output_tensor = torch.zeros(1, 256, 256, 256)

image_pairs = [(left_image, right_image), (left_image, right_image)]
output_tensors = [output_tensor, output_tensor]

train_dataset = StereoDataset(image_pairs, output_tensors)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = OccupancyNet('regnet', embed_dim=64, grid_size=16)
model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for (left_images, right_images), targets in train_loader:
        left_images, right_images, targets = left_images.to(device), right_images.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(left_images, right_images)
        
        # Compute loss
        loss = loss_function(predictions, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")