import keras
import numpy as np
import constants as c
from wav_reader import get_fft_spectrum
import matplotlib.pyplot as plt
import os

'''
读入训练文件列表 
FA_DIR：音频文件绝对路径前缀
path：训练文件列表
'''
def get_voxceleb1_datalist(FA_DIR, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(FA_DIR, string.split(",")[0]) for string in strings])
        labellist = np.array([int(string.split(",")[1]) for string in strings])
        f.close()
        audiolist = audiolist.flatten()
        labellist = labellist.flatten()
    return audiolist, labellist

'''
方法：计算不同帧数对应的最终输出尺寸是否大于 0
'''
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

'''
类：训练数据的生成器
'''
class DataGenerator(keras.utils.Sequence):
    def __init__(self,list_IDs, labels,dim,max_sec, step_sec, frame_step,batch_size=2,n_classes=1251, shuffle=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_sec = max_sec
        self.step_sec = step_sec
        self.frame_step = frame_step
        self.buckets = build_buckets(self.max_sec,self.step_sec,self.frame_step)

        self.on_epoch_end()

    def __getitem__(self, index):
        '返回一个 batch_size 的数据'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp =[self.list_IDs[k] for k in indexes]
        batch_data,batch_labels = self._gene_Data(list_IDs_temp,indexes)
        return batch_data,batch_labels

    def __len__(self):
        '计算有多少个 batch_size'
        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def on_epoch_end(self):
        '每次迭代后打乱训练列表'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _gene_Data(self, list_IDs_temp, indexes):
        '得到频谱数组和类标签，以输入模型进行训练'
        b_data = np.empty((self.batch_size,) + self.dim)
        b_labels = np.empty((self.batch_size,),dtype=int)

        for i,ID in enumerate(list_IDs_temp):
            b_data[i,:,:,0] = get_fft_spectrum(ID,self.buckets)
            b_labels[i] = self.labels[indexes[i]]

        b_labels = keras.utils.to_categorical(b_labels, num_classes=self.n_classes)
        b_labels = b_labels.reshape(self.batch_size,1,1,self.n_classes)

        # os.system("pause")

        return b_data,b_labels

'''绘制训练损失图像'''
def draw_loss_img(history_dict):
    loss_values = history_dict['loss']  # 训练损失
    # val_loss_values = history_dict['val_loss']  # 验证损失
    ep = range(1, len(loss_values) + 1)

    plt.plot(ep, loss_values, 'b', label="Training loss")  # bo表示蓝色原点
    # plt.plot(ep, val_loss_values, 'b', label="Validation loss")  # b表示蓝色实线
    plt.title("Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

'''绘制训练精度图像'''
def draw_acc_img(history_dict):
    accs = history_dict['acc']  # 训练精度
    # val_acc = history_dict['val_acc']  # 验证精度
    ep = range(1, len(accs) + 1)
    plt.plot(ep, accs, 'b', label="Training Acc")  # bo表示蓝色原点
    # plt.plot(ep, val_acc, 'b', label="Validation Acc")  # b表示蓝色实线
    plt.title("Train Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()  # 绘图
    plt.show()
