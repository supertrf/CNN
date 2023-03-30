# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:14:31 2023

@author: super
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def load_process_data(path):
    img_raw, labels = [], []
    # load data by folder
    for folder in path:
        img_path = glob.glob(folder+'/*')
        this_img = cv2.imread(img_path[0]) # read image
        this_img = cv2.medianBlur(this_img, 5) # apply filter to remove noise
        this_img = cv2.resize(this_img,(500,500),interpolation=cv2.INTER_LINEAR) # make sure all images are same size
        img_raw.append(this_img.astype(np.float32)) # convert to float32 dtype
        labels.append(img_path[0][-7:-5])
    labels = [label.replace('\\',' ') for label in labels] # replace slash with empty in labels
    return img_raw, labels



### Import Data 
train_folders = glob.glob('C:/Users/super/Downloads/Q2/train/*')
train_folders.sort()
test_folders = glob.glob('C:/Users/super/Downloads/Q2/test/*')
test_folders.sort()


train_img, train_labels = load_process_data(train_folders)
test_img, test_labels = load_process_data(train_folders)


# Normalize pixel values to be between 0 and 1
train_images = np.array([x for x in train_img]) / 255.0
test_images = np.array([x for x in test_img])  / 255.0
train_labels = np.array([int(x) for x in train_labels]).T
test_labels = np.array([int(x) for x in test_labels]).T


# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(11, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy is:", test_acc)
