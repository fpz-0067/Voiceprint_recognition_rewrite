import numpy as np
import csv
import time

const_train_list = "a_iden/train_list_iden.txt"
const_test_list = "a_iden/test_for_iden.txt"
save_list = "a_iden/new_train.csv"

origin_train_list = []
origin_test_list = []
write_list = []
with open(const_train_list, "r") as f:

    origin_train_list = f.readlines()
    xxx = origin_train_list
    for i in range(len(origin_train_list)):
        origin_train_list[i] = origin_train_list[i].strip('\n')  #去掉列表中每一个元素的换行符
    xxx = origin_train_list
    for i in range(len(origin_train_list)):
        origin_train_list[i] = origin_train_list[i].split(",")[0]
    f.close()

with open(const_test_list, "r") as f:

    origin_test_list = f.readlines()
    for i in range(len(origin_test_list)):
        origin_test_list[i] = origin_test_list[i].strip('\n')  #去掉列表中每一个元素的换行符
        origin_test_list[i] = origin_test_list[i].split(",")[0]
    f.close()

for i in range(len(origin_train_list)):
    ad = True
    for j in range(len(origin_test_list)):
        if(origin_train_list[i] == origin_test_list[j]):
            ad = False
            break
    if ad:
        write_list.append(xxx[i])

with open(save_list,'a',newline='') as f:
    csv_writer = csv.writer(f)
    # 添加标题
    csv_head = ['filename', 'speaker']
    csv_writer.writerow(csv_head)

    for l in write_list:
        # print(l)
        # time.sleep(3000)
        content = [l]
        csv_writer.writerow(content)

    f.close()