import os
import random

# Function to generate matched pairs for an identity
def generate_matched_pair(identity_images):
    pairs = []
    if len(identity_images) > 1:
        pairs.append((identity_images[0], identity_images[1]))
    return pairs

# Function to generate unmatched pairs for an identity
def generate_unmatched_pairs(identity_images, all_images, identities, n=3):
    pairs = []
    for _ in range(n):
        other_identity = random.choice(identities)
        other_identity_images = os.listdir(os.path.join(lfw_dir, other_identity))
        image1 = random.choice(identity_images)
        image2 = random.choice(other_identity_images)
        pairs.append((image1, other_identity, image2))
    return pairs

# Path to the directory containing LFW dataset
lfw_dir = "/media/statlab/SeagateHDD/Fateme Tavakoli/Face-Recognition-using-Siamese-Neural-Networks-main/data/ds/lfw2/Casia_webface/Casia_with_glass"

# List all identities
identities = os.listdir(lfw_dir)

# Shuffle identities
random.shuffle(identities)

# Calculate split points
matched_split_point = int(len(identities) * 0.2)
unmatched_split_point = int(len(identities) * 0.2)

# Divide identities into train and test sets
train_identities = identities[matched_split_point:]
test_identities = identities[:matched_split_point]

# Write pairs to train.txt and test.txt files
train_matched_pairs = []
train_unmatched_pairs = []
test_matched_pairs = []
test_unmatched_pairs = []

for identity in train_identities:
    identity_images = sorted(os.listdir(os.path.join(lfw_dir, identity)))
    matched_pair = generate_matched_pair(identity_images)
    if matched_pair:
        train_matched_pairs.append((identity, identity_images.index(matched_pair[0][0])+1, identity_images.index(matched_pair[0][1])+1))

    all_images = sorted(os.listdir(lfw_dir))
    negative_pairs = generate_unmatched_pairs(identity_images, all_images, identities)
    for pair in negative_pairs:
        other_identity = pair[1].split("/")[0]  # Extract the identity from the image path
        train_unmatched_pairs.append((identity, identity_images.index(pair[0])+1, other_identity, random.randint(1, len(os.listdir(os.path.join(lfw_dir, pair[1]))))))

for identity in test_identities:
    identity_images = sorted(os.listdir(os.path.join(lfw_dir, identity)))
    matched_pair = generate_matched_pair(identity_images)
    if matched_pair:
        test_matched_pairs.append((identity, identity_images.index(matched_pair[0][0])+1, identity_images.index(matched_pair[0][1])+1))

    all_images = sorted(os.listdir(lfw_dir))
    negative_pairs = generate_unmatched_pairs(identity_images, all_images, identities)
    for pair in negative_pairs:
        other_identity = pair[1].split("/")[0]  # Extract the identity from the image path
        test_unmatched_pairs.append((identity, identity_images.index(pair[0])+1, other_identity, random.randint(1, len(os.listdir(os.path.join(lfw_dir, pair[1]))))))

# Write pairs to train.txt
with open("train_casia_few.txt", "w") as f:
    f.write(f"Number of matched pairs: {len(train_matched_pairs)}\n")
    f.write(f"Number of unmatched pairs: {len(train_unmatched_pairs)}\n")
    f.write("Matched pairs:\n")
    for pair in train_matched_pairs:
        f.write(f"{pair[0]} {pair[1]} {pair[2]}\n")
    f.write("\nUnmatched pairs:\n")
    for pair in train_unmatched_pairs:
        f.write(f"{pair[0]} {pair[1]} {pair[2]} {pair[3]}\n")

# Write pairs to test.txt
with open("test_casia_few.txt", "w") as f:
    f.write(f"Number of matched pairs: {len(test_matched_pairs)}\n")
    f.write(f"Number of unmatched pairs: {len(test_unmatched_pairs)}\n")
    f.write("Matched pairs:\n")
    for pair in test_matched_pairs:
        f.write(f"{pair[0]}    {pair[1]}    {pair[2]}\n")
    f.write("\nUnmatched pairs:\n")
    for pair in test_unmatched_pairs:
        f.write(f"{pair[0]}    {pair[1]}    {pair[2]}    {pair[3]}\n")

print("train.txt and test.txt files generated successfully.")

# import os


# def rename_images(directory):
#     for root, dirs, files in os.walk(directory):
#         for i, file in enumerate(files, start=1):
#             filename, file_extension = os.path.splitext(file)
#             new_filename = f"{os.path.basename(root)}_{i:04d}{file_extension}"
#             os.rename(os.path.join(root, file), os.path.join(root, new_filename))

#             # If there are subdirectories, rename images in those directories too
#             for subdir in dirs:
#                 subdir_path = os.path.join(root, subdir)
#                 for j, sub_file in enumerate(os.listdir(subdir_path), start=1):
#                     sub_filename, sub_file_extension = os.path.splitext(sub_file)
#                     new_sub_filename = f"{subdir}_{j:04d}{sub_file_extension}"
#                     os.rename(os.path.join(subdir_path, sub_file), os.path.join(subdir_path, new_sub_filename))

# directory = "/media/statlab/SeagateHDD/Fateme Tavakoli/Face-Recognition-using-Siamese-Neural-Networks-main/data/ds/lfw2/Casia_webface/Casia_with_glass"
# rename_images(directory)