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
    input_shape=(224, 224, 3))
conv_base.trainable = False
#conv_base.summary()
model = Sequential()
model.add(conv_base)

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer=Adam(lr=0.0001), metrics=['accuracy'])

model.summary()

#training
training_set = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input,
#validation_split = 0.2,
)

test_set = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input,
rescale=1./255
)

train_generator = training_set.flow_from_directory(
    'dataset',
    target_size = (224,224),
    batch_size = 64,
    subset= 'training',
    seed=45,
    shuffle = True,
    class_mode = 'binary'
)
print(train_generator.class_indices)
'''
valid_generator = training_set.flow_from_directory('dataset',
    target_size= (224,224),
    shuffle=True,
    batch_size=10,
    subset = 'validation',
    seed=42,
    class_mode='binary',
 )'''

my_callbacks = [
    keras.callbacks.EarlyStopping(patience=5, monitor='accuracy'),
]

trained_model = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs = 10,
    #validation_data = valid_generator,
    validation_steps = 5,
    callbacks=my_callbacks
    )

model.save('tb_keras.h5')

model.save_weights('tb_keras_weights.h5')

#class activation maps
#grad-CAM