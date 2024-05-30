import os
import shutil

# Path to the folder containing the images
image_folder_path = '/media/statlab/SeagateHDD/Fateme Tavakoli/few/img_align_celeba/img_align_celeba'

# Path to the text file containing the identities
identity_file_path = '/media/statlab/SeagateHDD/Fateme Tavakoli/few/identity_CelebA.txt'

# Read the identity file and process it
with open(identity_file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    try:
        # Split the line to get the image filename and identity
        image_filename, identity = line.strip().split()
        
        # Construct full path for the source image
        src_image_path = os.path.join(image_folder_path, image_filename)
        
        # Check if the source image exists
        if not os.path.isfile(src_image_path):
            print(f"Warning: Source image '{src_image_path}' not found.")
            continue
        
        # Create a directory for the identity if it doesn't exist
        identity_dir = os.path.join(image_folder_path, identity)
        os.makedirs(identity_dir, exist_ok=True)
        
        # Define the destination path for the image
        dst_image_path = os.path.join(identity_dir, image_filename)
        
        # Copy the image to the destination directory
        shutil.copy(src_image_path, dst_image_path)
        print(f"Copied '{src_image_path}' to '{dst_image_path}'")
    except Exception as e:
        print(f"Error processing line '{line.strip()}': {e}")

print("Images have been organized into subfolders by identity.")

