
# # ### resize to 112*112
# import os
# import cv2

# # Path to the directory containing CelebA images
# celeba_dir = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/results_only_glass_n"

# # Output directory where resized images will be saved
# output_dir = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/results_only_glass_n"
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




# ## lfw backward to folder
# import os
# import shutil

# def organize_images(source_dir, destination_dir):
#     # Iterate through all files in the source directory
#     for filename in os.listdir(source_dir):
#         # Check if the file is an image
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             # Remove the _0001.jpg part from the filename to get the directory name
#             folder_name = os.path.splitext(filename)[0].rsplit('_', 1)[0]
#             # Create the corresponding directory structure in the destination directory if it doesn't exist
#             os.makedirs(os.path.join(destination_dir, folder_name), exist_ok=True)
#             # Copy the image to its corresponding directory in the destination directory
#             shutil.copy(os.path.join(source_dir, filename), os.path.join(destination_dir, folder_name, filename))

# if __name__ == "__main__":
#     # Specify the source directory containing images
#     destination_directory = "//media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/data"
#     # Specify the destination directory where images will be organized
#     source_directory = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/results_only_glass_n"

#     # Call the function to organize images into subdirectories based on their filenames
#     organize_images(source_directory, destination_directory)




# make few shots
import os
import random
import shutil

def count_images_in_directory(directory):
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return count

def delete_images_randomly(directory, max_images=1):
    image_list = os.listdir(directory)
    if len(image_list) > max_images:
        images_to_delete = len(image_list) - max_images
        images_to_delete = min(images_to_delete, len(image_list))  # Ensure not to delete more images than available
        images_to_delete_list = random.sample(image_list, images_to_delete)
        for image in images_to_delete_list:
            image_path = os.path.join(directory, image)
            os.remove(image_path)

def process_directories(main_directory, max_images=2):
    for subdir in os.listdir(main_directory):
        path = os.path.join(main_directory, subdir)
        if os.path.isdir(path):
            image_count = count_images_in_directory(path)
            if image_count > max_images:
                print(f'Deleting images in {subdir}...')
                delete_images_randomly(path, max_images)
                print(f'Deleted images in {subdir} to contain at most {max_images} photos.')

main_directory = '/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/Casia-webface/Casia_with_glass'
process_directories(main_directory, max_images=5)


# ###### mix up
# import os
# import shutil

# def merge_folders(folder1_path, folder2_path):
#     folder1_subfolders = [f.name for f in os.scandir(folder1_path) if f.is_dir()]
#     folder2_subfolders = [f.name for f in os.scandir(folder2_path) if f.is_dir()]

#     common_subfolders = list(set(folder1_subfolders) & set(folder2_subfolders))

#     for subfolder in common_subfolders:
#         subfolder1_path = os.path.join(folder1_path, subfolder)
#         subfolder2_path = os.path.join(folder2_path, subfolder)

#         for filename in os.listdir(subfolder2_path):
#             old_filepath = os.path.join(subfolder2_path, filename)
#             new_filename = filename.split('.')[0] + "_aug" + os.path.splitext(filename)[1]
#             new_filepath = os.path.join(subfolder2_path, new_filename)
#             os.rename(old_filepath, new_filepath)

#         for filename in os.listdir(subfolder2_path):
#             shutil.move(os.path.join(subfolder2_path, filename), os.path.join(subfolder1_path, filename))

#         os.rmdir(subfolder2_path)

#     print("Folders merged successfully")


# folder1_path = "ds/Casia-webface/datasets"
# folder2_path = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/Casia-webface/datasets_orginal_out_res"

# merge_folders(folder1_path, folder2_path)
