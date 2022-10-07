import os

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from PIL import Image
np.random.seed(11) 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier

folder_benign = '../input/data/data/benign'
folder_malignant = '../input/data/data/malignant'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Load Data
ims_benign = [read(os.path.join(folder_benign, filename)) for filename in os.listdir(folder_benign)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant, filename)) for filename in os.listdir(folder_malignant)]
X_malignant = np.array(ims_malignant, dtype='uint8')



y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

# Merge data and shuffle it
X = np.concatenate((X_benign, X_malignant), axis = 0)
y = np.concatenate((y_benign, y_malignant), axis = 0)
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
y = y[s]



X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.20, random_state=11)

X_train_scaled = np.zeros(X_train.shape)
X_test_scaled = np.zeros(X_test.shape)

X_train_scaled -= X_train.mean(axis=(1, 2), keepdims=True)
X_train_scaled /= X_train.std(axis=(1, 2), keepdims=True)

X_test_scaled -= X_test.mean(axis=(1, 2), keepdims=True)
X_test_scaled /= X_test.std(axis=(1, 2), keepdims=True)


# Set the CNN model 
input_shape = (75, 100, 3)

def create_cnn(lr= 0.001, init= 'normal', activ= 'relu'):

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation= activ,padding = 'Same',input_shape=input_shape))
    model.add(Conv2D(32,kernel_size=(3, 3), activation= activ,padding = 'Same',))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation= activ,padding = 'Same'))
    model.add(Conv2D(64, (3, 3), activation= activ,padding = 'Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.40))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    model.summary()
    
    optimizer = Adam(lr=lr,
                     beta_1=0.9,
                     beta_2=0.999,
                     epsilon=None,
                     decay=0.0,
                     amsgrad=False)
    
    model.compile(optimizer = optimizer ,
                  loss = "binary_crossentropy",
                  metrics=["accuracy"])

    return model

# Create sci-kit learn API
dnn = KerasClassifier(build_fn= create_cnn, verbose=0)


# With data augmentation to prevent overfitting 
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train_scaled)




#Gridsearch Parameters
activ = ['tanh', 'relu']
batch_size = [10]
epochs = [100]
lr = [1e-1,1e-2,1e-3,1e-4]

parameters = dict(activ = activ,
                  epochs = epochs,
                  batch_size = batch_size,
                  lr = lr
                 )

# Fit the model
kfold = KFold(n_splits=3, 
              random_state=11)

grid = GridSearchCV(dnn , 
                    parameters, 
                    cv=kfold,
                    scoring= 'accuracy')

grid_result = grid.fit(X_train_scaled, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



	
