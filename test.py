from PIL import Image
import glob
import numpy as np

import torch
from torchvision import transforms

from occupancynet import *

# Define a custom transform to handle alpha channel
def remove_alpha_channel(img):
    return img.convert('RGB') if img.mode == 'RGBA' else img

model = OccupancyNet('regnet', embed_dim=64, grid_size=16)
model.load_state_dict(torch.load('best.pt', weights_only=True))
model.eval()  # Set the model to evaluation mode

# Left and right images of stereo cameras, and 3D occupancy maps
left_image_root = "blender_stereo_images/left_images/"
right_image_root = "blender_stereo_images/right_images/"
paths = glob.glob("blender_stereo_images/occupancy_grids/*.npy")

# Image transformation to tensor, resize to match your network's expected input size
transform = transforms.Compose([
    transforms.Lambda(remove_alpha_channel),
    transforms.Resize((256, 256)),  # Resize to match your network's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

image_pairs = []
output_tensors = []

for i, path in enumerate(paths):

    filename = "image_" + str(i)

    # Load left image
    left_image_path = f"{left_image_root}{filename}.png"  # Assuming images are PNG
    with Image.open(left_image_path) as img:
        left_image = transform(img)
    
    # Load right image
    right_image_path = f"{right_image_root}{filename}.png"
    with Image.open(right_image_path) as img:
        right_image = transform(img)
    
    # Load occupancy grid (output tensor)
    output_tensor = torch.from_numpy(np.load(path)).unsqueeze(0)  # Add channel dimension
    output_tensor = output_tensor.float()

    # Ensure output_tensor has the correct shape if needed
    if output_tensor.shape != (1, 32, 32, 32):

        raise ValueError(f"Expected output tensor shape (1, 32, 32, 32), got {output_tensor.shape}")

    image_pairs.append((left_image, right_image))
    output_tensors.append(output_tensor)

for (left_image, right_image), output_tensor in zip(image_pairs, output_tensors):
    # Predict occupancy map
    occupancy_map = model(left_image.unsqueeze(0), right_image.unsqueeze(0))

    # Remove batch and channel dimensions for element-wise comparison
    occupancy_map = occupancy_map.squeeze()
        
    # Convert to binary if they are not already (assuming a threshold of 0.5)
    if not occupancy_map.dtype == torch.bool:
        occupancy_map = (occupancy_map > 0.5).float()
    if not output_tensor.dtype == torch.bool:
        output_tensor = (output_tensor > 0.5).float()
    
    # Compute accuracy
    correct_predictions = (occupancy_map == output_tensor).sum().item()
    total_voxels = occupancy_map.numel()
    accuracy = correct_predictions / total_voxels
    
    print(f"Accuracy for this pair: {accuracy:.4f}")