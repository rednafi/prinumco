import os 
from glob import glob

# making directories
def directory_generator():
    for i in range(10):
        if not os.path.exists("train"):
            os.mkdir('train')

        if not os.path.exists('test'):
            os.mkdir('test')
            
        if not os.path.exists('train/'+ str(i)):
            os.mkdir('train/'+ str(i))
            
        if not os.path.exists('test/'+ str(i)):
            os.mkdir('test/' + str(i))
            
        else:
            pass


# checking the fonts
def check_fonts():

    """
    Make a directory named 'custom' in the '/usr/share/fonts/truetype' path and copy the bangla fonts there. 
    """
    path = '/usr/share/fonts/truetype/custom/'
    files = [f for f in glob(path + "**/*.ttf", recursive=True)]

    return files