# import os 
# from glob import glob
# from PIL import Image
# from tqdm import tqdm
# import numpy as np
# import os

# def binarize_image(image_path, threshold = 128):
#     img = Image.open(image_path)
#     fn = lambda x: 255 if x > threshold else 0
#     bw = img.convert('L').point(fn, mode = '1')
#     bw.save(image_path)

# def train_test_image_binarize(train_folder, test_folder):

#     # train image binarization
#     train_file_paths = []
#     for root, dirs, files in os.walk(train_folder):
#         for file in files:
#             if file.endswith(".jpg"):
#                 train_file_paths.append(os.path.join(root, file))

#     for f in tqdm(train_file_paths):
#         binarize_image(f)

#     # test image binarization
#     test_file_paths = []
#     for root, dirs, files in os.walk(test_folder):
#         for file in files:
#             if file.endswith(".jpg"):
#                 test_file_paths.append(os.path.join(root, file))


#     for f in tqdm(test_file_paths):
#         binarize_image(f)