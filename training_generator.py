import pandas as pd
import os
import shutil

'''
FILE_PATH = 'train.csv'
IMAGES_PATH = 'train'
TARGET_DIR = 'dataset/positive'
'''
def image_class_extractor(file_path, images_path, target_dir, target):
    df = pd.read_csv(file_path)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        print('dir created')

    count = 0
    for (i, row) in df.iterrows():
        if row['LABEL'] == target:
            filename = row['ID']+'.png'
            image_path = os.path.join(images_path, filename)
            image_copy_path = os.path.join(target_dir, filename)
            shutil.copy2(image_path, image_copy_path)
            print('Moving image', filename)
            count+=1
    print(count, 'images moved to', target_dir)

#image_class_extractor('train.csv','train','dataset/positive', 1)
#image_class_extractor('train.csv','train','dataset/negative', 0)

