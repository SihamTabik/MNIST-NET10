import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras import optimizers
from  keras.preprocessing.image import random_rotation, random_shear, random_shift, random_zoom
import tensorflow as tf
from keras.models import load_model
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.datasets import mnist
import csv
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#Function to apply random rotations, shears and shifts
def augment_data(dataset, dataset_labels, augmentation_factor=1, use_random_rotation=True, use_random_shear=True, use_random_shift=True, use_random_zoom=True):
        augmented_image = []
        augmented_image_labels = []

        for num in range (0, dataset.shape[0]):
                # original image:
                augmented_image.append(dataset[num])
                augmented_image_labels.append(dataset_labels[num])

                for i in range(0, augmentation_factor):

                        if use_random_rotation:
                                augmented_image.append(random_rotation(dataset[num], 20, row_axis=0, col_axis=1, channel_axis=2))
                                augmented_image_labels.append(dataset_labels[num])

                        if use_random_shear:
                                augmented_image.append(random_shear(dataset[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
                                augmented_image_labels.append(dataset_labels[num])

                        if use_random_shift:
                                augmented_image.append(random_shift(dataset[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
                                augmented_image_labels.append(dataset_labels[num])

        s = np.arange(len(augmented_image))#Permutation
        return np.array(augmented_image)[s], np.array(augmented_image_labels)[s]

def crop(img, size,isRand=True):#Function to crop an image
  n = len(img) - size
  x_0 = 2
  y_0 = 2
	if isRand:
	        x_0 = random.randint(0,n)
	        y_0 = random.randint(0,n)

  croped_img = img[x_0:x_0+size,y_0:y_0+size]
  return croped_img

#Function to apply rotation. A modification of Keras random_rotation.code
#https://github.com/NVIDIA/keras/blob/master/keras/preprocessing/image.py
def rotation_2(x,theta, row_axis=1,col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.):
  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

  h, w = x.shape[row_axis], x.shape[col_axis]
  transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
  x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
  return x

#Function to apply to a dataset rotation given in angles. 
def rotations(dataset, dataset_labels, angles):
        augmented_image = []
        augmented_image_labels = []

        for num in range (0, dataset.shape[0]):
                # original image:
                augmented_image.append(dataset[num])
                augmented_image_labels.append(dataset_labels[num])

                for theta in angles:
                        augmented_image.append(rotation_2(dataset[num], theta , row_axis=0, col_axis=1, channel_axis=2))
                        augmented_image_labels.append(dataset_labels[num])

        return np.array(augmented_image), np.array(augmented_image_labels)


#Function to aaply random elastic transformations. A modification of the chsasank code elastic_transform.py.
#https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

#Funtion to apply elastic deformation to all a dataset 
def getElastics(images, labels,alpha, sigma):
        augmented = []
        augmented_labels = []
        for i in range(len(images)):
                news = np.array([elastic_transform(images[i][0], alpha,sigma) for j in range(4)])
                news = np.append(news, images[i][0])
                augmented.append(news)
                augmented_labels.append([labels[i], labels[i], labels[i], labels[i], labels[i]])

        return np.reshape(augmented, newshape=(5 * len(images), images.shape[2], images.shape[3]  ) ) , np.reshape(augmented_labels, 5 * len(images))

#Function to apply data augmentation process
def augmentate(images, labels, alpha, sigma):
        images = images.reshape(images.shape[0], 1,24,24)
        deformated, new_labels = getElastics(images, labels,alpha, sigma)
        deformated = deformated.reshape(deformated.shape[0], 24, 24, 1)
        augmented = rotations(deformated, new_labels,[-16,-8, 8, 16])
        return augmented[0].reshape(augmented[0].shape[0],24,24,1), augmented[1]

#Function to apply data augmentation process
def augmentate_2(images, labels, alpha, sigma):
        images = images.reshape(images.shape[0], 1,24,24)
        deformated, new_labels = getElastics(images, labels,alpha, sigma)
        deformated = deformated.reshape(deformated.shape[0], 24, 24,1)
        return deformated, new_labels

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Building datasets
X_train = X_train[0:60000]
y_train = y_train[0:60000]

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = np.array([crop(X_train[i],24) for i in range(len(X_train))  ])
X_test = np.array([crop(X_test[i],24,isRand=False) for i in range(len(X_test)) ])

X_train_org = X_train
Y_train_org = np_utils.to_categorical(y_train,10)

X_train = np.array([X_train[i] - np.mean(X_train[i]) for i in range(len(X_train))])
X_test = np.array([X_test[i] - np.mean(X_test[i]) for i in range(len(X_test))])

#X_train_elastic, y_train_elastic = augmentate(X_train, y_train,6,4)
#Y_train_elastic = np_utils.to_categorical(y_train_elastic,10)

X_train_rand_el, y_train_rand_el = augmentate_2(X_train, y_train,6,4)
Y_train_rand_el = np_utils.to_categorical(y_train_rand_el,10)


X_train_random, y_train_random = augment_data(X_train, y_train,augmentation_factor = 1, use_random_rotation=True, use_random_shear=True, use_random_shift=True)
X_train_2, y_train_2 = augmentate_2(X_train_random, y_train_random,6,4)
Y_train_2 = np_utils.to_categorical(y_train_2,10)

#X_train_enc, y_train_enc = augment_data(X_train, y_train,use_random_rotation=True, use_random_shear=False, use_random_shift=False)
#X_train_enc, y_train_enc = augment_data(X_train_enc, y_train_enc,use_random_rotation=False, use_random_shear=True, use_random_shift=False)
#X_train_enc, y_train_enc = augment_data(X_train_enc, y_train_enc,use_random_rotation=False, use_random_shear=False, use_random_shift=True)

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
#Y_train_random = np_utils.to_categorical(y_train_random,10)
#Y_train_enc = np_utils.to_categorical(y_train_enc,10)

#Function to train neural net given train set
def experiment(X_train, Y_train, file):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum = 0.95,  nesterov=True)
	model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

        print("Start training")

        model.fit(X_train, Y_train,  batch_size=128, nb_epoch=50, verbose=1)


        print("Start evaluation...")
        score = model.evaluate(X_test, Y_test, verbose=0)

        print(score)

        preds = model.predict( X_test, batch_size=None, verbose=0)

        preds = np.argmax(preds,1)

        print(preds.shape)
        wrong = 0
        mistakes = []

        for i in range(len(preds)):
                if preds[i]!=y_test[i]:
                        wrong+=1
                        mistakes = np.append(mistakes, i)
        err = []
        print("Number of errors:")
        print(wrong)
        print(mistakes)
        err = np.append(err, wrong)

        print(err)

        model.save(file)



print("ENTRENAMIENTO ORIGINAL")

#experiment(X_train_org, Y_train_org, './models_org/modelo1_60.h5')
#experiment(X_train_org, Y_train_org, './models_org/modelo2_60.h5')
#experiment(X_train_org, Y_train_org, './models_org/modelo3_60.h5')
#experiment(X_train_org, Y_train_org, './models_org/modelo4_60.h5')
#experiment(X_train_org, Y_train_org, './models_org/modelo5_60.h5')


print("ENTRENAMIENTO ALEATORIO")
#experiment(X_train_random, Y_train_random, './models_random_noRep/modelo1_60.h5')
#experiment(X_train_random, Y_train_random, './models_random_noRep/modelo2_60.h5')
#experiment(X_train_random, Y_train_random, './models_random_noRep/modelo3_60.h5')
#experiment(X_train_random, Y_train_random, './models_random_noRep/modelo4_60.h5')
#experiment(X_train_random, Y_train_random, './models_random_noRep/modelo5_60.h5')

print("ENTRENAMIENTO ENCADENADO")

#experiment(X_train_enc, Y_train_enc, './models_random_encadenado/modelo1_60.h5')
#experiment(X_train_enc, Y_train_enc, './models_random_encadenado/modelo2_60.h5')
#experiment(X_train_enc, Y_train_enc, './models_random_encadenado/modelo3_60.h5')
#experiment(X_train_enc, Y_train_enc, './models_random_encadenado/modelo4_60.h5')
#experiment(X_train_enc, Y_train_enc, './models_random_encadenado/modelo5_60.h5')

print("ENTRENAMIENTO CON ROTACIONES Y DEFORMACIONES ELASTICAS:")

#experiment(X_train_elastic, Y_train_elastic, './models_elastic/modelo1.h5')
#experiment(X_train_elastic, Y_train_elastic, './models_elastic/modelo2.h5')
#experiment(X_train_elastic, Y_train_elastic, './models_elastic/modelo3.h5')
#experiment(X_train_elastic, Y_train_elastic, './models_elastic/modelo4.h5')
#experiment(X_train_elastic, Y_train_elastic, './models_elastic/modelo5.h5')


print("ENTRENAMIENTO ALEATORIO Y DEFORMACIONES ELASTICAS:")

experiment(X_train_2, Y_train_2, './models_random_elastic_2/modelo1.h5')
experiment(X_train_2, Y_train_2, './models_random_elastic_2/modelo2.h5')
experiment(X_train_2, Y_train_2, './models_random_elastic_2/modelo3.h5')
experiment(X_train_2, Y_train_2, './models_random_elastic_2/modelo4.h5')
experiment(X_train_2, Y_train_2, './models_random_elastic_2/modelo5.h5')