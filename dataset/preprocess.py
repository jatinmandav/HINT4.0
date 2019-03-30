import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter

image_size = (48, 48)

def image_preprocess(image):
    image = image.astype('float32')
    image = image/255.

    return image

def read_data(path, type='Training'):
    df = pd.read_csv(path)
    print(df.describe())
    #print(df.head())

    assert type == 'Training' or type == 'PrivateTest' or type == 'PublicTest', 'Please Provide Training/PrivateTest/PublicTest to read the data'

    df = df[df['Usage'] == type]
    #print(df.head())

    pixels = df['pixels'].tolist()
    (width, height) = image_size

    faces = []
    for pixel in tqdm(pixels, desc='Generating {} Data'.format(type)):
        face = [int(pix) for pix in pixel.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(df['emotion']).as_matrix()

    return faces, emotions


if __name__ == '__main__':
    faces, emotions = read_data('fer2013/fer2013.csv')
    print(emotions)
