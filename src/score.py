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

'''
方法：计算不同帧数对应的最终输出尺寸是否大于 0
'''


def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step)# 1s/10ms = 100
	end_frame = int(max_sec*frames_per_sec)# 10s*100 = 1000
	step_frame = int(step_sec*frames_per_sec)# 1*100 = 100
	for i in range(0, end_frame+1, step_frame):# [100,...,1000]
		s = i # 100,200,300,...,1000
		s = np.floor((s-7+2)/2) + 1  # conv1  np.floor()返回不大于输入参数的最大整数,向下取整
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets


def get_embeddings_from_list_file(model, list_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	# print(result)
	# time.sleep(10)
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(c.FA_DIR+x, buckets)) # 获取每个人的语音特征 矩阵
	# print(result['features'])
	# time.sleep(10)
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))# to input (1, 512, 1000, 1) per voice(feature)
	# print(result['embedding'])
	# time.sleep(10)
	return result[['filename','speaker','embedding']]


'''
方法：输入同一个人的不同语音，计算输出向量之间的余弦夹角，得到评分
'''


def get_id_result(weights):
	# print("Loading model weights from [{}]....".format(weights))
	# model = vggvox_model()
	# model.load_weights(weights)

	model = load_model(c.MODEL_LOAD_PATH)

	model = Model(inputs=model.layers[0].input,outputs=model.layers[34].output) # 取 fc7 层的输出（1024,）

	print("Processing enroll samples....")
	enroll_result = get_embeddings_from_list_file(model, c.ENROLL_LIST_FILE, c.MAX_SEC)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	# print(enroll_embs)
	# time.sleep(10)
	# print("enroll_embs: ",enroll_embs.shape)# [3,1024]
	speakers = enroll_result['speaker']
	# print("speakers: ",speakers.shape)# [3,]
	# print(speakers)# [19,13,27] three class

	print("Processing test samples....")
	test_result = get_embeddings_from_list_file(model, c.TEST_LIST_FILE, c.MAX_SEC)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])
	# print(test_embs)
	# time.sleep(10)

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers) # 计算余弦相似度（ 夹角，越小越好 ）
	print("distances: ")
	print(distances)
	time.sleep(5)

	scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	# scores = pd.concat([scores, distances],axis=1)
	# scores['result'] = scores[speakers].idxmin(axis=1)  # 取最小值第一次出现的索引

	scores['result'] = distances[speakers].idxmin(axis=1) # 取最小值第一次出现的索引 （改动：仅输出结果，删除余弦相似度）
	scores['correct'] = (scores['result'] == scores['test_speaker']) # bool to int
	res = scores['correct'].value_counts(1)
	print(res)
	# print("scores: ")
	# print(scores)
	# os.system("pause")

	print("Writing outputs to [{}]....".format(c.RESULT_FILE))
	result_dir = os.path.dirname(c.RESULT_FILE)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	with open(c.RESULT_FILE, 'w') as f:
		scores.to_csv(f, index=False)



if __name__ == '__main__':
	time_start = time.time()
	get_id_result(c.PERSONAL_WEIGHT)
	time_end = time.time()
	print('cost: ', time_end - time_start)
