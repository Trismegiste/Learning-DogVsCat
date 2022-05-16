# HELLOWORLD
# https://youtu.be/iJ_ZTWsHmB4
# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

targets = []
features = []

files = glob.glob('/home/trismegiste/Images/datasets/DogNCat/*.jpg')
random.shuffle(files)

for file in files[:500]:
    features.append(np.array(Image.open(file).resize((75, 75))))
    target = [1, 0] if "cat" in file else [0, 1]
    targets.append(target)

features = np.array(features)
targets = np.array(targets)

print("Shape features", features.shape)
print("Shape targets", targets.shape)

#for a in [random.randint(0, len(targets)) for _ in range(10)]:
#    plt.imshow(features[a])
#    plt.show()
    
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.1, random_state=42)

print("X_train", X_train.shape)
print("X_valid", X_valid.shape)
print("y_train", y_train.shape)
print("y_valid", y_valid.shape)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Placeholder
x = tf.placeholder(tf.float32, (None, 75, 75, 3), name="Image")
y = tf.placeholder(tf.float32, (None, 2), name="Targets")
