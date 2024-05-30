import os
import cv2

# Define the directory to start from
start_directory = '/media/statlab/SeagateHDD/Fateme Tavakoli/Face-Recognition-using-Siamese-Neural-Networks-main/data/ds/lfw2/Casia_webface/datasets'


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
