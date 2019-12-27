import os
import time
os.environ["MKL_NUM_THREADS"] = '12'
os.environ["NUMEXPR_NUM_THREADS"] = '12'
os.environ["OMP_NUM_THREADS"] = '12'
import numpy as np
from keras.models import load_model
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
from model import vggvox_model
from wav_reader import get_fft_spectrum
from tools import build_buckets
import constants as c
from keras.models import Model

'''
使用EER作为测试指标
'''


def score(testfile,fa_data_dir,test_model_path,max_sec, step_sec, frame_step,metric):
    print("Use {} for test".format(testfile))

    verify_list = np.loadtxt(testfile, str)

    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(fa_data_dir, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(fa_data_dir, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    # Load model
    print("Load model form {}".format(test_model_path))
    model = load_model(test_model_path)
    model = Model(inputs=model.layers[0].input, outputs=model.layers[34].output)  # 取 fc7 层的输出（1024,）

    print("Start testing...")
    total_length = len(unique_list) # 4715
    feats, scores, labels = [], [], []
    buckets = build_buckets(max_sec, step_sec, frame_step)
    for c, ID in enumerate(unique_list):
        if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
        specs = get_fft_spectrum(ID,buckets)
        v = model.predict(specs.reshape(1,*specs.shape,1))
        feats += [v]

    feats = np.array(feats)

    # 计算相似度
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1,0,0]
        v2 = feats[ind2,0,0]

        distances = cdist(v1, v2, metric=metric)
        print(distances[0][0])

        if c>-1:
            break

score(c.NEW_TEST_FILE,c.FA_DIR,c.TEST_MODEL_PATH,c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP,c.COST_METRIC)