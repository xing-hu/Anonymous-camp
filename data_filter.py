import glob
from PIL import Image
import numpy as np
import shutil

white_threshold = 200
keep_threshold = 0.5


def white(x):
    return x > white_threshold


data_path = '/Users/menrui/Downloads/img_align_celeba/'
output_path = '../clean_celeba/'
files = glob.glob(data_path + '*.*')
clean_files = 0
file_counts = 0
for file in files:
    image = Image.open(file)
    image = np.asarray(image)
    image = image.mean(axis=-1).flatten().tolist()

    white_count = len(list(filter(white, image)))
    file_counts += 1
    if white_count / len(image) > keep_threshold:
        clean_files += 1
        shutil.copy(file, output_path + str(clean_files) + '.jpg')
    if file_counts % 100 == 0:
        print(file_counts, clean_files)
