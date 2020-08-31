import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras import backend as K

tb_model = load_model('tb_keras.h5')
print(tb_model.summary())



test_set = image.ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)
test_generator = test_set.flow_from_directory(
    'test',
    target_size = (224,224),
    batch_size = 10,
    shuffle = False,
    class_mode = None
)

ans = tb_model.predict(test_generator)
#print(test_generator)

