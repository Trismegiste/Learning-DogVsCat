# HELLOWORLD

# https://youtu.be/iJ_ZTWsHmB4
# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

targets = []
features = []

files = glob.glob('dataset/Cat/*.jpg')
for file in files:
    features.append(Image.open(file).resize((75, 75)))
    target = [1, 0]
    targets.append(target)

files = glob.glob('dataset/Dog/*.jpg')
for file in files:
    features.append(Image.open(file).resize((75, 75)))
    target = [0, 1]
    targets.append(target)

features = np.array(features)
targets = np.array(targets)

print("Shape features", features.shape)
print("Shape targets", targets.shape)