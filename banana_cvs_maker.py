import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import cv2


DATADIR = './data/imgs'
X_FILE = 'features.npy'
Y_FILE = 'labels.npy'
IMG_SIZE = 50
BANANA_FOLDERS = ['Banana', 'Banana_Lady_Finger', 'Banana_Red']

train_file = open("data/train_labels.csv", "w")
test_file = open("data/test_labels.csv", "w")
train_file.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
test_file.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')


def create_dataset():
    for folder in BANANA_FOLDERS:
        path = os.path.join(DATADIR, folder)
        for img in os.listdir(path):
            if np.random.uniform(0,1) < 0.3:
                test_file.write(os.path.join(path, img) + ',100,100,banana,0,0,100,100\n')
            else:
                train_file.write(os.path.join(path, img) + ',100,100,banana,0,0,100,100\n')

create_dataset()
train_file.close()
test_file.close()