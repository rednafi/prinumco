import os 
from glob import glob

# making directories
def directory_generator():
    # ensuring that the directories exist
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    for i in range(10):
        ensure_dir('dataset/train/' + str(i) + '/')
        ensure_dir('dataset/test/'+ str(i) + '/')


# checking the fonts
def check_fonts():

    """
    Make a directory named 'custom' in the '/usr/share/fonts/truetype' path and copy the bangla fonts there. 
    """
    path = '/usr/share/fonts/truetype/custom/'
    files = [f for f in glob(path + "**/*.ttf", recursive=True)]

    return files
