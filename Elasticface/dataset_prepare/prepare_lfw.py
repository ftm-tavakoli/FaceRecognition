# # num check
# import os

# def count_images_in_directory(directory):
#     count = 0
#     for _, _, files in os.walk(directory):
#         count += len(files)
#     return count

# def get_directories_with_more_than_n_images(main_directory, n=1):
#     count = 0
#     result = {}
#     for subdir in os.listdir(main_directory):
#         path = os.path.join(main_directory, subdir)
#         if os.path.isdir(path):
#             image_count = count_images_in_directory(path)
#             if image_count > n:
#                 result[subdir] = image_count
#                 count += 1
#     return count, result

# main_directory = '/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/LFW/one_shot_lfw'
# count, result = get_directories_with_more_than_n_images(main_directory, n=1)

# print(f'Total directories with more than 2 images: {count}')
# for subdir, image_count in result.items():
#     print(f'{subdir}: {image_count} images')


############################################################################

#make few shot
# import os
# import random
# import shutil

# def count_images_in_directory(directory):
#     count = 0
#     for _, _, files in os.walk(directory):
#         count += len(files)
#     return count

# def delete_images_randomly(directory, max_images=1):
#     image_list = os.listdir(directory)
#     if len(image_list) > max_images:
#         images_to_delete = len(image_list) - max_images
#         images_to_delete = min(images_to_delete, len(image_list))  # Ensure not to delete more images than available
#         images_to_delete_list = random.sample(image_list, images_to_delete)
#         for image in images_to_delete_list:
#             image_path = os.path.join(directory, image)
#             os.remove(image_path)

# def process_directories(main_directory, max_images=1):
#     for subdir in os.listdir(main_directory):
#         path = os.path.join(main_directory, subdir)
#         if os.path.isdir(path):
#             image_count = count_images_in_directory(path)
#             if image_count > max_images:
#                 print(f'Deleting images in {subdir}...')
#                 delete_images_randomly(path, max_images)
#                 print(f'Deleted images in {subdir} to contain at most {max_images} photos.')

# main_directory = "/media/statlab/SeagateHDD/Fateme Tavakoli/ElasticFace/ds/LFW/one_shot+Augmentation"
# process_directories(main_directory, max_images=2)


