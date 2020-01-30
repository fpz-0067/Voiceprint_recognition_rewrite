import numpy as np
import constants as c
import os

def get_voxceleb1_datalist(FA_DIR, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(FA_DIR, string.split(",")[0]) for string in strings])
        labellist = np.array([int(string.split(",")[1]) for string in strings])
        f.close()
    # return audiolist, labellist
    print(audiolist)
    print(audiolist.flatten())


# 调用方法
# get_voxceleb1_datalist(c.FA_DIR,"D:\Python_projects/vggvox_rewrite\a_veri/trainlist.txt")

# x = np.ones((3,5),dtype=int)
# print("origin: ")
# print(x)
# print(x.shape)
# x = x.reshape(3,1,1,5)
# print("after: ")
# print(x)
# print(x.shape)

# i = 1
# print(i ==1)
# print(i is 1)

import os
import time
import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

from model import vggvox_model
from wav_reader import get_fft_spectrum
import constants as c
from keras.models import Model
from keras.backend import set_learning_phase

# model = vggvox_model()
# model.load_weights(c.PERSONAL_WEIGHT)

# set_learning_phase(1)

# model = load_model(c.MODEL_LOAD_PATH)
#
# model = Model(inputs=model.layers[0].input,outputs=model.layers[34].output) #34
#
# x = np.random.normal(0,1,(512,100))
# print(x.shape)
# xxx = model.predict(x.reshape(1,*x.shape,1))
# print(xxx)
# print(xxx.shape)

s = 200  # 100,200,300,...,1000
s = np.floor((s - 7 + 2) / 2) + 1  # conv1  np.floor()返回不大于输入参数的最大整数,向下取整
s = np.floor((s - 3) / 2) + 1  # mpool1
s = np.floor((s - 5 + 2) / 2) + 1  # conv2
s = np.floor((s - 3) / 2) + 1  # mpool2
s = np.floor((s - 3 + 2) / 1) + 1  # conv3
s = np.floor((s - 3 + 2) / 1) + 1  # conv4
s = np.floor((s - 3 + 2) / 1) + 1  # conv5
s = np.floor((s - 3) / 2) + 1  # mpool5
s = np.floor((s - 1) / 1) + 1  # fc6
print(s)



