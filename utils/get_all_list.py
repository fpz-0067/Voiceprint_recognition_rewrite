import csv
import os
import os.path

father_path = "F:/Vox_data/vox1_dev_wav/wav/"
sub_path = ""
write_path = "D:\Python_projects/vggvox_rewrite/a_iden/tmp.csv"
with open(write_path,'a',newline='') as f:
    csv_writer = csv.writer(f)
    # 添加标题
    csv_head = ['filename', 'speaker']
    csv_writer.writerow(csv_head)

    real_i = 0

    for i in range(1,1252):

        done = 0

        if(i>= 1000):
            sub_path = "id1" + str(i)
        elif(i>=100):
            sub_path = "id10" + str(i)
        elif(i>=10):
            sub_path = "id100" + str(i)
        else:
            sub_path = "id1000" + str(i)
        # print(sub_path)

        # father_path+sub_path # F:/Vox_data/vox1_dev_wav/wav/id10001
        id_path = father_path+sub_path
        filelist = os.listdir(id_path) # 获取 id10001 下的所有文件

        for filename in filelist:
            all_vox_path = os.path.join(id_path, filename)
            all_vox_list = os.listdir(all_vox_path)

            for vox_filename in all_vox_list:
                this_vox_path = os.path.join(all_vox_path, vox_filename)
                this_vox_path = this_vox_path.replace("\\", "/")
                this_vox_path = this_vox_path.replace("F:/Vox_data/vox1_dev_wav/wav/", "")

                # 添加内容
                csv_content = [this_vox_path, i - 1]
                csv_writer.writerow(csv_content)

                real_i = real_i+1

    # 关闭文件
    f.close()
    print(real_i)