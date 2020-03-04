import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
# %matplotlib inline

# Reading the train and test csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())

# Number of images in the training and the test datasets.
print('There are {} images in the train dataset.'.format(train.shape[0]))
print('There are {} images in the test datasets.'.format(test.shape[0]))

# Number of unique animals in the datasets
print('There are {} unique animals in the dataset.'.format(train.Animal.unique().shape[0]))

print('The dataset contain images of the following animals:\n\n', train.Animal.unique())

animal_count = pd.value_counts(train.Animal)

plt.figure(figsize=(15, 8))
ax = sns.barplot(x=animal_count.index, y=animal_count.values)
ax.set_xticklabels(ax.get_xticklabels(), rotation= 90)
ax.set_title('Frequency Distribution of the Animals in the Training Data')
ax.set(xlabel='Animals', ylabel='Count')
# plt.show()

### Checking for any null values in the labels
labels  =  train.Animal
labels.isnull().any()

### Train adn Test image folder paths

TRAIN_PATH = 'train/train/'
TEST_PATH = 'test/test/'

### Importing Python Image Library and Opencv library
from PIL import Image
import cv2

print('Animal: ',train.Animal[10])
Image.open(TRAIN_PATH + train.Image_id[10])

img = cv2.imread(TRAIN_PATH+train.Image_id[0])

print(img)

### Displaying the image dimensions of the first five images in the training dataset.
sample = train.head()
for idx in sample.Image_id:
    img = cv2.imread(TRAIN_PATH + idx)
    print('{} : {}'.format(idx, img.shape))

### Creating a function to resize the images in the training data.
from tqdm import tqdm


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    return img


#temp = train.sample(frac=0.3)
#train = temp.reset_index(drop=True)

train_img = []
for img_path in tqdm(train.Image_id.values):
    train_img.append(read_img(TRAIN_PATH + img_path))


### Displaying the image dimensions of the first five images in the training dataset.
sample = train.head()
for idx in sample.Image_id:
    img = cv2.imread(TRAIN_PATH + idx)
    print('{} : {}'.format(idx, img.shape))

import gc

# Convert the image data into an array.
# Since the range of color(RGB) is in the range of (0-255).
# Hence by dividing each image by 255, we convert the range to (0.0 - 1.0)

X_train = np.array(train_img, np.float32) / 255

del train_img
gc.collect()

mean_img = X_train.mean(axis = 0)
std_dev = X_train.std(axis = 0)

X_norm = (X_train - mean_img)/ std_dev
X_norm.shape

del X_train
gc.collect()



