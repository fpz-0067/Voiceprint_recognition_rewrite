import os
os.environ["MKL_NUM_THREADS"] = '12'
os.environ["NUMEXPR_NUM_THREADS"] = '12'
os.environ["OMP_NUM_THREADS"] = '12'
import time
import numpy as np
from keras.models import load_model
from wav_reader import get_fft_spectrum
from tools import build_buckets
import constants as c

'''
说话人识别
使用Acc作为指标
'''


def iden(testfile,fa_data_dir,iden_model,max_sec, step_sec, frame_step):
    # 读入测试数据、标签
    print("Use {} for test".format(testfile))

    iden_list = np.loadtxt(testfile, str,delimiter=",")

    labels = np.array([int(i[1]) for i in iden_list])
    voice_list = np.array([os.path.join(fa_data_dir, i[0]) for i in iden_list])

    # Load model
    print("Load model form {}".format(iden_model))
    model = load_model(iden_model)

    print("Start identifying...")
    total_length = len(voice_list)
    res, p_labels = [], []
    buckets = build_buckets(max_sec, step_sec, frame_step)
    for c, ID in enumerate(voice_list):
        if c % 1000 == 0: print('Finish identifying for {}/{}th wav.'.format(c, total_length))
        specs = get_fft_spectrum(ID, buckets)
        v = model.predict(specs.reshape(1, *specs.shape, 1))
        v = np.squeeze(v)
        p_labels.append(np.argmax(v))

    p_labels = np.array(p_labels)
    compare = (labels == p_labels)
    counts = sum(compare==True)
    acc = counts/total_length
    print(acc)

iden(c.IDEN_TEST_FILE,c.FA_DIR,c.IDEN_MODEL_LOAD_PATH,c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)