# import os
# import cv2

# # Path to the directory containing CelebA images
# celeba_dir = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/Casia-webface/datasets_orginal"

# # Output directory where resized images will be saved
# output_dir = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/Casia-webface/datasets_orginal"

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Function to resize images
# def resize_images(input_dir, output_dir, desired_size):
#     # Iterate through each image in the directory
#     for filename in os.listdir(input_dir):
#         if filename.endswith(".jpg"):
#             # Read the image
#             img_path = os.path.join(input_dir, filename)
#             img = cv2.imread(img_path)
            
#             # Resize the image to the desired size
#             resized_img = cv2.resize(img, desired_size)
            
#             # Save the resized image to the output directory
#             output_path = os.path.join(output_dir, filename)
#             cv2.imwrite(output_path, resized_img)

# # Desired size for resizing images
# desired_size = (112, 112)

# # Resize images in CelebA directory and save them to the output directory
# resize_images(celeba_dir, output_dir, desired_size)

# print("Images resized and saved to:", output_dir)

import os
import cv2

# Define the directory to start from
start_directory = '/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/Casia-webface/datasets'


# Define the target size
target_size = (112, 112)

# Walk through the directory
for dirpath, dirnames, filenames in os.walk(start_directory):
    for filename in filenames:
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full file path
            filepath = os.path.join(dirpath, filename)
            
            # Read the image using OpenCV
            img = cv2.imread(filepath)
            
            # Check if the image was successfully read
            if img is not None:
                # Resize the image
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # Save the resized image, overwrite the original
                cv2.imwrite(filepath, img_resized)
                
                print(f"Resized {filename} in {dirpath} to {target_size}")
            else:
                print(f"Could not read {filename} in {dirpath}")




