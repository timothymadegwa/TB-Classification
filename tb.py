import numpy as np
import matplotlib.pyplot as pyplot
import keras
from keras import backend as K
from keras.models import Sequential
from keras.applications import VGG16
from keras.layers import Activation, Conv2D, MaxPool2D, Dropout
from keras.layers.core import Dense, Flatten
from keras.preprocessing import image
from keras.optimizers import Adam


#CNN
conv_base = VGG16(weights='imagenet',
    include_top=False,
    input_shape=(448, 448, 3))
conv_base.trainable = False

model = Sequential()
model.add(conv_base)
'''
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(448,448,3)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
'''
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.summary()

#training
training_set = image.ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    rotation_range=50,
    featurewise_center = True,
    featurewise_std_normalization = True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.25,
    zoom_range=0.1,
    zca_whitening = True,
    channel_shift_range = 20,
    vertical_flip = True ,
    validation_split = 0.2,
    fill_mode='constant'
)
test_set = image.ImageDataGenerator(rescale=1./255)

train_generator = training_set.flow_from_directory(
    'dataset',
    target_size = (448,448),
    batch_size = 32,
    subset= 'training',
    seed=45,
    shuffle = True,
    class_mode = 'binary'
)
print(train_generator.class_indices)

valid_generator = training_set.flow_from_directory('dataset',
    target_size= (448,448),
    shuffle=True,
    batch_size=10,
    subset = 'validation',
    seed=42,
    class_mode='binary',
 )

trained_model = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs = 10,
    validation_data = valid_generator,
    validation_steps = 5
    )

model.save('tb_keras.h5')

model.save_weights('tb_keras_weights.h5')

#class activation maps
#grad-CAM