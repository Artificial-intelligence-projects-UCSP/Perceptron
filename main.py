from keras.datasets import mnist
import matplotlib.pyplot as plt
# cargar (descargar si es necesario) el conjunto de datos MNIST
(x_training, y_training), (X_evaluation, y_evaluation) = mnist.load_data()
# plot 4 imagenes en escala de grises
plt.subplot(221)
plt.imshow(x_training[8], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_training[13], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_training[22], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_training[5], cmap=plt.get_cmap('gray'))
plt.show()

print(X_evaluation.shape)
print(x_training.shape)
import numpy as np 
import pandas as pd 
import os
import copy
import time
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import copy
import time
import albumentations as A
import torch_optimizer as optim
from res_mlp_pytorch import ResMLP
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
