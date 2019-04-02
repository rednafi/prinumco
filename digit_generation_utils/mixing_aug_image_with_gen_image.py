import os
import shutil
from tqdm import tqdm

#moving the augmented images to the corresponding folders 
def copytree(source_path, destination_path, symlinks=False, ignore=None):
    
    print('Moving augmented images to the corresponding folders...')
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for item in tqdm(os.listdir(source_path)):
        source = os.path.join(source_path, item)
        destination = os.path.join(destination_path, item)
        if os.path.isdir(source):
            copytree(source, destination, symlinks, ignore)
        else:
            if not os.path.exists(destination) or os.stat(source).st_mtime - os.stat(destination).st_mtime > 1:
                shutil.copy2(source, destination)

#removing the original augmented folder
def remove_output(output_folder_path):
    print('Removing redundant folders...')
    shutil.rmtree(output_folder_path)