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
# get_voxceleb1_datalist(c.FA_DIR,"D:\Python_projects/vggvox_rewrite\cfg/trainlist.txt")

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



