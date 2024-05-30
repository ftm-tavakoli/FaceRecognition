import os
import shutil
import torch
import piq
from PIL import Image
from torchvision import transforms

# Define paths
lfw_path = 'path_to_lfw+'
output_path = 'path_to_output_high_quality'
num_images_per_identity = 5  # Change this to the desired number of images per identity

# Prepare the output directory
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Define image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Function to compute BRISQUE score
def compute_brisque(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    score = piq.brisque(image, data_range=1.0).item()
    return score

# Iterate over each identity
for identity in os.listdir(lfw_path):
    identity_path = os.path.join(lfw_path, identity)
    if os.path.isdir(identity_path):
        image_scores = []
        
        # Iterate over each image for the identity and compute BRISQUE score
        for image_name in os.listdir(identity_path):
            image_path = os.path.join(identity_path, image_name)
            score = compute_brisque(image_path)
            image_scores.append((image_path, score))
        
        # Sort images by BRISQUE score in ascending order
        image_scores.sort(key=lambda x: x[1])
        
        # Keep the top num_images_per_identity images
        selected_images = image_scores[:num_images_per_identity]
        
        # Copy the selected images to the output directory
        output_identity_path = os.path.join(output_path, identity)
        if not os.path.exists(output_identity_path):
            os.makedirs(output_identity_path)
        for image_path, _ in selected_images:
            shutil.copy(image_path, os.path.join(output_identity_path, os.path.basename(image_path)))

print('High-quality image selection complete!')
