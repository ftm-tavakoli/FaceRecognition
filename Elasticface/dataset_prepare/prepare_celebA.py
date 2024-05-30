"""
Fist make whole celebA dataset like lfw and make it 1 shot
Second make the augmented dataset like the orginal one and few shot
lastly mix then using kharkari
"""



# ########make it like LFW and FEW shot ##########
# import os
# import random
# from collections import defaultdict
# import shutil

# celebA_folder = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/celebA/results_celeb"

# identity_file = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/celebA/Anno/identity_CelebA.txt"

# identities = defaultdict(list)

# with open(identity_file, 'r') as file:
#     for line in file:
#         image, identity = line.strip().split()
#         identities[int(identity)].append(image)

# selected_folder = os.path.join("/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/celebA", "CelebA_few_aug")
# os.makedirs(selected_folder, exist_ok=True)

# identity_labels_file = os.path.join(selected_folder, "identity_labels.txt")
# with open(identity_labels_file, 'w') as labels_file:
#     for identity, images in identities.items():
#         if len(images) >= 2:
#             selected_images = random.sample(images, 1)
#             identity_folder = os.path.join(selected_folder, f"identity_{identity}")
#             os.makedirs(identity_folder, exist_ok=True)
#             for image in selected_images:
#                 source_path = os.path.join(celebA_folder, image)
#                 destination_path = os.path.join(identity_folder, image)
#                 shutil.copyfile(source_path, destination_path)
#                 labels_file.write(f"{image}: {identity}\n")




# ### put all image in one directory #####

# import os
# import shutil

# def collect_images(source_dir, destination_dir):
#     # Create the destination directory if it doesn't exist
#     if not os.path.exists(destination_dir):
#         os.makedirs(destination_dir)

#     # Iterate through all subdirectories in the source directory
#     for root, dirs, files in os.walk(source_dir):
#         for filename in files:
#             # Check if the file is an image (you can add more extensions if needed)
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                 # Get the full path of the image
#                 source_file = os.path.join(root, filename)
#                 # Move the image to the destination directory
#                 shutil.move(source_file, destination_dir)

# if __name__ == "__main__":
#     # Specify the source directory containing subfolders with images
#     source_directory = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/CelebA/few_shot_celebA"
#     # Specify the destination directory where all images will be collected
#     destination_directory = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/CelebA/few_shot_celebA_all"

#     # Call the function to collect images
#     collect_images(source_directory, destination_directory)




# ###khar kari
import os
import shutil

def merge_folders(folder1_path, folder2_path):
    folder1_subfolders = [f.name for f in os.scandir(folder1_path) if f.is_dir()]
    folder2_subfolders = [f.name for f in os.scandir(folder2_path) if f.is_dir()]

    common_subfolders = list(set(folder1_subfolders) & set(folder2_subfolders))

    for subfolder in common_subfolders:
        subfolder1_path = os.path.join(folder1_path, subfolder)
        subfolder2_path = os.path.join(folder2_path, subfolder)

        for filename in os.listdir(subfolder2_path):
            old_filepath = os.path.join(subfolder2_path, filename)
            new_filename = filename.split('.')[0] + "_aug" + os.path.splitext(filename)[1]
            new_filepath = os.path.join(subfolder2_path, new_filename)
            os.rename(old_filepath, new_filepath)

        for filename in os.listdir(subfolder2_path):
            shutil.move(os.path.join(subfolder2_path, filename), os.path.join(subfolder1_path, filename))

        os.rmdir(subfolder2_path)

    print("Folders merged successfully")


folder1_path = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/Casia-webface/datasets_orginal_out_res"
folder2_path = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/LFW/one_shot_lfw"

merge_folders(folder1_path, folder2_path)

