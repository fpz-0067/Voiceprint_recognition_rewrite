import os
import random
import time

import numpy as np
import pandas as pd
from keras import optimizers
from model import vggvox_model
from wav_reader import get_fft_spectrum
import constants as c


# 方法：计算不同帧数对应的最终输出尺寸是否大于 0
def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)  # 1s/10ms = 100
    end_frame = int(max_sec * frames_per_sec)  # 10s*100 = 1000
    step_frame = int(step_sec * frames_per_sec)  # 1*100 = 100
    for i in range(0, end_frame + 1, step_frame):  # [100,...,1000]
        s = i  # 100,200,300,...,1000
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1  np.floor()返回不大于输入参数的最大整数,向下取整
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


# 获取语音文件的频谱 --> 数组
# def get_train_list(train_list_file):
# 	buckets = build_buckets(c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)
#     read_in = pd.read_csv(tra)
# 	signal = get_fft_spectrum(wav_file, buckets)
# 	embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape,1)))
# 	return embedding

# 使用one-hot编码将标签向量化
def to_one_hot(label,dimension=1251):
    results = np.zeros(dimension)
    results[label]=1.
    return results

# 方法：返回 np 训练数据
def get_np_list(file_list,buckets):
    voice = np.empty()
    for pt in range(len(file_list)):
        np.concatenate(get_fft_spectrum(c.FA_DIR+pt, buckets))
    print(voice.shape)

# 方法：获取训练数据
def get_train_list(path):
    buckets = build_buckets(c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)
    read_csv = pd.read_csv(path, delimiter=",")
    print("Preprocessing voice data...")
    read_csv["voice"] = read_csv["filename"].apply(lambda x: get_fft_spectrum(c.FA_DIR+x, buckets))
    read_csv["lable"] = read_csv["speaker"].apply(lambda x: to_one_hot(x - 1))

    return read_csv

# fit_generator
def gene(train_data):
    length = len(train_data["voice"])
    list = random.sample(range(length), length)
    while 1:
        for i in range(length):
            yield (train_data["voice"][list[i]],train_data["lable"][list[i]])

# 方法：训练
def train_vggvox_model(train_list_file):
    model = vggvox_model()
    train_data = get_train_list(train_list_file)
    # 编译模型
    model.compile(optimizer=optimizers.RMSprop(lr=0.1),
                  loss="categorical_crossentropy",# 使用分类交叉熵作为损失函数
                  metrics=['acc'])  # 使用精度作为指标

    # 测试输入格式 (**Most important**)
    # data = np.random.randn(1,512,30,1)
    # lable = np.zeros((1251,))
    # lable[1000] = 1.
    # lable = lable.reshape((1,1,1,1251))
    # print(lable.shape)

    train_data["voice"] = train_data["voice"].apply(lambda x: x.reshape(1,*x.shape,1))
    train_data["lable"] = train_data["lable"].apply(lambda x: x.reshape((1, 1, 1, 1251)))

    print("Start training...")
    history = model.fit_generator(gene(train_data),
                                  epochs=100,
                                  steps_per_epoch=c.TRAIN_NUM)
    model.save_weights(filepath=c.PERSONAL_WEIGHT)

    print("loss: ",min(history.history["loss"]))
    print("Done!")

# 测试方法：训练
train_vggvox_model(c.TRAIN_LIST_FILE)