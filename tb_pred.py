import keras
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
from keras import backend as K

tb_model = load_model('model.13-0.92.h5')
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


submission = pd.read_csv('SampleSubmission.csv')

submission = submission.sort_values(by='ID')
submission.LABEL = ans

submission.to_csv('submission.csv', index=False, sep=',')

