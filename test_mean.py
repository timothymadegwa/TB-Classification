from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = pd.read_csv('Test.csv')

names = names['ID'].values
means = []

for name in names:
   image = Image.open('test/test/'+name+'.png')
   image = np.asarray(image).astype('float64')
   mean = image.mean(axis = (0,1), dtype='float64')

   means.append(mean)
arr = np.array(means)
print(arr.mean())
print(arr.std())

plt.hist(arr, bins=100)
plt.show()
