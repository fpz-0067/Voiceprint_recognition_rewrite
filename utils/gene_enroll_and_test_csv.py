import numpy as np
import csv

# 定义常量
const_source_iden_split_file = "/home/longfuhui/all_data/iden_split.txt"
new_enroll_list_path = "/home/longfuhui/shengwenshibie/vggvox_rewrite/cfg/tmp.csv"
new_test_list_path = "D:/Python_projects/vggvox-speaker-identification-master/cfg/new_test_list.csv"
how_many_id = 64
how_many_record_per_id = 1

# 方法：处理源文本 1
def get_posi_and_id_1(id_num,diff = True):
    path = const_source_iden_split_file
    f = open(path)
    line = f.readline()
    c = 0
    pre = 0
    list = []
    while line:
        id = int(line[5:9])
        posi = line[2:-1]
        if(diff):
            if(id != pre):
                pre = id
                c += 1
                list.append([posi, id])
        else:
            c += 1
            list.append([posi,id])
        line = f.readline()
        if(c >= id_num):
            f.close()
            break

    return list

# 方法：处理源文本 2
def get_posi_and_id_2(id_num,line_num_per_id):
    path = const_source_iden_split_file
    f = open(path)
    line = f.readline()
    c = 0
    pre = 0
    lnpi = line_num_per_id
    list = []
    while line:
        id = int(line[5:9])
        posi = line[2:-1]
        if(id != pre):
            c += 1
            if (c > id_num):
                f.close()
                break
            pre = id
            lnpi = line_num_per_id
            line = f.readline()
            continue
        else:
            if(lnpi > 0):
                list.append([posi,id])
                lnpi -= 1
        line = f.readline()
    return list

# 方法：处理 val_split.txt
def get_posi_and_id_3():
    path = const_source_iden_split_file
    f = open(path)
    line = f.readline()
    list = []
    while line:
        id = int(line[5:9])
        posi = line[2:-1]
        list.append([posi, id])
        line = f.readline()

    return list

# 方法：创建注册CSV
def crete_enroll_csv(path):
    with open(path,'a',newline='') as f:
        csv_writer = csv.writer(f)
        # 添加标题
        csv_head = ['filename', 'speaker']
        csv_writer.writerow(csv_head)
        # 添加内容
        list = get_posi_and_id_1(how_many_id)
        for l in list:
            csv_content = [l[0], l[1]]
            csv_writer.writerow(csv_content)

        # 关闭文件
        f.close()

# 方法：创建测试CSV
def crete_test_csv(path):
    with open(path,'a',newline='') as f:
        csv_writer = csv.writer(f)
        # 添加标题
        csv_head = ['filename', 'speaker']
        csv_writer.writerow(csv_head)
        # 添加内容
        list = get_posi_and_id_2(how_many_id,how_many_record_per_id)
        for l in list:
            csv_content = [l[0], l[1]]
            csv_writer.writerow(csv_content)

        # 关闭文件
        f.close()

# 方法：创建测试CSV
def crete_test_csv_new(path):
    with open(path,'a',newline='') as f:
        csv_writer = csv.writer(f)
        # 添加标题
        csv_head = ['filename', 'speaker']
        csv_writer.writerow(csv_head)
        # 添加内容
        list = get_posi_and_id_3()
        for l in list:
            csv_content = [l[0], l[1]]
            csv_writer.writerow(csv_content)
        # 关闭文件
        f.close()


# 调用方法：获取注册语音文件
crete_enroll_csv(new_enroll_list_path)

# 调用方法：获取测试语音文件
# crete_test_csv(new_test_list_path)

# 调用方法：获取新的注册语音文件
# crete_test_csv_new(new_test_list_path)