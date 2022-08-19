# import libraries
from segmentation_model import get_model
from classification_model import classification_model
from coef_and_loss import dice_coef, dice_loss
import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow as tf


# read our data
train = pd.read_csv('train_ship_segmentations_v2.csv')

# decode pixels to take mask
def find_mask(encoded_pixels, size):
    my_img = []

    for i in range(0, len(encoded_pixels), 2):
        steps = encoded_pixels[i+1]
        start = encoded_pixels[i]

        pos_of_pixels = [start+j for j in range(steps)]
        my_img.extend(pos_of_pixels)

    mask_img = np.zeros((size**2), dtype=np.uint8)
    mask_img[my_img] = 1
    mask = np.reshape(mask_img, (size,size)).T

    return mask


# random our dataset
np.random.seed(0)
np.random.shuffle(train.values)


# split data into 2 groups: with ships and without
train_without_ship = train[train['EncodedPixels'].isna()]
train_without_ship.index = [i for i in range(len(train_without_ship))]
train_with_ship = train[train['EncodedPixels'].notna()].groupby('ImageId')['EncodedPixels'].apply(lambda x: ' '.join(x)).to_frame()
train_with_ship = train_with_ship.reset_index()



# count of images
n = 3000

# take arrays of images and coordinates, for those images
imgs_to_classification = []
imgs_to_segmentation = []
mask_to_segmentation = []
y = []


for i in range(n):
    try:
        # read and resize image
        img = cv.imread('train_v2/'+train_with_ship['ImageId'][i])
        img = cv.resize(img, (256,256))
        img = img.astype(np.uint8)

        # decode pixels and take mask
        encoded_pixels = [int(k) for k in train_with_ship['EncodedPixels'][i].split()]
        mask = find_mask(encoded_pixels, 768)
        mask = cv.resize(mask, (256,256))
        imgs_to_segmentation.append(img)
        mask_to_segmentation.append(mask)

        # take 50% of images with ships and 50% without ships
        if i % 2 == 0:
            imgs_to_classification.append(img)
            y.append(np.array([1,0]))
        else:
            img = cv.imread('train_v2/'+train_without_ship['ImageId'][i])
            img = cv.resize(img, (256,256))
            img = img.astype(np.uint8)
            imgs_to_classification.append(img)
            y.append(np.array([0,1]))

    except:
        # Corrupted img
        pass

# change dtypes of our input and output data
imgs_to_classification = np.array(imgs_to_classification, dtype=np.uint8)
y = np.array(y, dtype=np.uint8)
imgs_to_segmentation = np.array(imgs_to_segmentation, dtype=np.float16)
mask_to_segmentation = np.array(mask_to_segmentation, dtype=np.float16)



# compiling two models
model = get_model((256,256))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=dice_loss, metrics=[dice_coef])

cnn3 = classification_model()
cnn3.compile(loss='binary_crossentropy', optimizers='adam', metrics=['accuracy'])



from sklearn.model_selection import train_test_split
# splitting data into train and valid groups
X_segm_train, X_segm_valid, y_segm_train, y_segm_valid = train_test_split(imgs_to_segmentation, mask_to_segmentation.reshape(-1,256,256,1), test_size=0.1, random_state=42)
X_class_train, X_class_valid, y_class_train, y_class_valid = train_test_split(imgs_to_classification, y, test_size=0.2, random_state=42)


# fitting model
cnn3.fit(X_class_train, y_class_train, validation_data=(X_class_valid, y_class_valid), epochs=40, batch_size=32)
model.fit(X_segm_train, y_segm_train, validation_data=(X_segm_valid, y_segm_valid), epochs=100, batch_size=32)
