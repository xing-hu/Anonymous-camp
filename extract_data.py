import glob
import shutil
import os
import numpy as np

os.makedirs('./data/train_A/')
os.makedirs('./data/test_A/')
files_people = glob.glob('../clean_celeba/*.*')
np.random.shuffle(files_people)
for file in files_people[:7000]:
    shutil.copy(file, './data/train_A')
for file in files_people[7000:]:
    shutil.copy(file, './data/test_A')


#files_kartoon = glob.glob('kartoon/*.*')
#np.random.shuffle(files_kartoon)
#for file in files_kartoon[:8000]:
#    shutil.copy(file, './data/train_B')
#for file in files_kartoon[8000:]:
#    shutil.copy(file, './data/test_B')