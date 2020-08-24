import numpy as np
import matplotlib.pyplot as pyplot
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPool2D, Dropout
from keras.layers.core import Dense, Flatten
from keras.preprocessing import image


#CNN

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()

#training
training_set = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
)
test_set = image.ImageDataGenerator(rescale=1./255)

train_generator = training_set.flow_from_directory(
    'dataset',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)
print(train_generator.class_indices)

trained_model = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs = 10,

)
#class activation maps
#grad-CAM