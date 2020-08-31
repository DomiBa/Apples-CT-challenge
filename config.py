import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import socket
from enum import Enum

##### Data #####
batch_size = 1
num_channels = 1
num_slices = 1
num_classes_seg = 1

views = 50
dense_views = 50
train_dim1 =  512
train_dim2 = views
train_dim1 = 1376
train_dim2 = 50
test_dim1 = train_dim1
test_dim2 = train_dim2
train_input_shape = [train_dim1, train_dim2, num_channels]

rot_array = np.linspace(0, -np.pi, dense_views)