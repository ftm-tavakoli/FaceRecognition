import os

folder_path = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/CelebA/results_only_glass_n"


files = os.listdir(folder_path)

for filename in files:
    if filename.endswith(".jpg"):  
        new_filename = filename[:-11]  
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        print(f"Renamed {filename} to {new_filename}")
