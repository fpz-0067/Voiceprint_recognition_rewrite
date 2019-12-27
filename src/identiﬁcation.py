import os
os.environ["MKL_NUM_THREADS"] = '12'
os.environ["NUMEXPR_NUM_THREADS"] = '12'
os.environ["OMP_NUM_THREADS"] = '12'
import time
import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

from model import vggvox_model
from wav_reader import get_fft_spectrum
from tools import build_buckets
import constants as c
from keras.models import Model

'''
说话人身份确认
'''


def iden(testfile,fa_data_dir,iden_model):
    # 读入测试数据、标签
    print("Use {} for test".format(testfile))

    iden_list = np.loadtxt(testfile, str)

    labels = np.array([int(i[1]) for i in iden_list])
    voice_list = np.array([os.path.join(fa_data_dir, i[0]) for i in iden_list])

    # Load model
    print("Load model form {}".format(iden_model))
    model = load_model(iden_model)


iden(c.TEST_LIST_FILE,c.FA_DIR,c.IDEN_MODEL_LOAD_PATH)