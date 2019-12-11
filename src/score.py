import os
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

from model import vggvox_model
from wav_reader import get_fft_spectrum
import constants as c
from keras.models import Model

# 方法：计算不同帧数对应的最终输出尺寸是否大于 0
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


# def get_embedding(model, wav_file, max_sec):
# 	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
# 	signal = get_fft_spectrum(wav_file, buckets)
# 	embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape,1)))
# 	return embedding


# def get_embedding_batch(model, wav_files, max_sec):
# 	return [ get_embedding(model, wav_file, max_sec) for wav_file in wav_files ]


def get_embeddings_from_list_file(model, list_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(c.FA_DIR+x, buckets)) # 获取每个人的语音特征 矩阵
	# print(result['features'])
	# print("********")
	# print(result['features'].shape)# shape: [3,512,1000]
	# print("********")
	# print(result['features'][0].shape)
	# os.system("pause")
	# print("***features[0]*****")
	# print(result['features'][0])# shape: [512,1000] or [512,900]
	# print("********")
	# print(result['features'][0].shape)
	# print(result['features'][1].shape)
	# print(result['features'][2].shape)
	# print("****") # 900
	# xxx = result['features'].apply(lambda x: x.reshape(1,*x.shape,1))
	# x = result['features']
	# xxx = x.reshape(1,*x.shape,1)
	# print(xxx.shape)
	# os.system("pause")

	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))# to input (1, 512, 1000, 1) per voice(feature)
	# print(result['embedding'])# shape: [3,1024]
	# print(result['embedding'][0])# shape: [1024,]
	# print(result['embedding'][0].shape)
	# print(result['embedding'][1].shape)
	# print(result['embedding'][2].shape)
	return result[['filename','speaker','embedding']]


# 方法：输入同一个人的不同语音，计算输出向量之间的余弦夹角，得到评分
def get_id_result():
	print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(c.WEIGHTS_FILE)
	model = Model(inputs=model.layers[0].input,outputs=model.layers[34].output) # 取 fc7 层的输出（1024,）
	# model.summary()
	# os.system("pause")

	print("Processing enroll samples....")
	enroll_result = get_embeddings_from_list_file(model, c.ENROLL_LIST_FILE, c.MAX_SEC)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	# print(enroll_embs)
	# print("enroll_embs: ",enroll_embs.shape)# [3,1024]
	speakers = enroll_result['speaker']
	# print("speakers: ",speakers.shape)# [3,]
	# print(speakers)# [19,13,27] three class

	print("Processing test samples....")
	test_result = get_embeddings_from_list_file(model, c.TEST_LIST_FILE, c.MAX_SEC)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])
	# print(test_embs)
	# os.system("pause")

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers) # 计算余弦相似度（ 夹角，越小越好 ）
	# print("distances: ")
	# print(distances)

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
	get_id_result()
	time_end = time.time()
	print('cost: ', time_end - time_start)
