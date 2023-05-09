""" program for fashion product classification .  This program i copied from youtube
to learn and understand keras / easy use using kagel kernel.

Code is mainly written for jupyter notebook/kernel/
So there are lots of mistake if i run here ...
Dataset "fashion-mnist-dataset_test_train" are doenloaded from kagel
This is just a copy """



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np              # linear algebra
import pandas as pd             # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import tensorflow as tf
print(tf.__version__)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import matplotlib.pyplot as plt
#%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



data_train_file = "../input/fashion-mnist-dataset-test-train/fashion-mnist_test.csv"
data_test_file = "../input/fashion-mnist-dataset-test-train/fashion-mnist_train.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)

df_train.head()

def get_feature_lebels(df):
    #delect all colums but the first
    features = df.values[:, 1:]/255
    #The first column is the label. Conveniently called "label"
    labels = df["labels"].values
    return features, labels


train_features, train_labels = get_features_labels(df_train)
test_features, test_labels = get_features_labels(df_test)


print(train_features.shape)
print(train_labels.shape)


train_features[20,300:320]


example_index = 221
plt.figure()
_ = plt.imshow(np.reshape(train_features[example_index, :], (28,28)), "gray")



train_labels.shape

train_labels[example_index]

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

train_labels.shape

train_labels[example_index]


""" This is the most important part where build model keras"""

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#we will now compile and build the summery of the model

model.compile(loss="categorical_crossentropy",
             optimizer="rmsprop",
             metrics=["accuracy"])

model.summary()




# training with keras

EPOCHS=2
BATCH_SIZE=128



model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)


test_loss, test_acc = model.evaluate(test_features,test_labels)


print("test_acc:", test_acc)



