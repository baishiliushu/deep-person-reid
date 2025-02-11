import os
import random

##输入数据集路径
directory_path = '/home/spring/test_qzj/data/data20241224/data_all'
#输出txt文件路径
txt_path = '/home/spring/test_qzj/data/data20241224'
Query_Test_Ratio = 0
Query_Ratio = 0
import os

# 把三个文件夹的ID合成一个

def get_files_in_folders(folder_list):
    all_files = []
    for folder in folder_list:
        # 检查文件夹是否存在
        if os.path.isdir(folder):
            # 获取文件夹内所有文件并保存绝对路径
            files = [os.path.join(folder, file) for file in os.listdir(folder)]
            all_files.extend(files)
        else:
            print(f"Warning: {folder} does not exist or is not a directory.")
    return all_files
def count_and_sample_subfolders(directory, sample_ratio=Query_Test_Ratio):
    # 获取目录下的所有子文件夹
    folders = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
    # 计算要抽取的文件夹数量
    sample_count = max(0, int(len(folders) * sample_ratio))
    print(sample_count)
    # 随机选择指定比例的文件夹
    sampled_folders = random.sample(folders, sample_count)

    # 训练集和测试集/查询集的划分
    train_folders = [folder for folder in folders if folder not in sampled_folders]
    print(f"总文件夹数: {len(folders)}")
    print(f"随机选择的文件夹({sample_ratio * 100}%): {sampled_folders}")

    return train_folders, sampled_folders


train, test_query = count_and_sample_subfolders(directory_path)
print(len(train))
# 创建带摄像头子目录的文件路径列表
## train
cam0_file = [os.path.join(directory_path, folder, 'normal') for folder in train]
# half：采集的图片中只有半身或坐、蹲姿势下的图片
cam1_file = [os.path.join(directory_path, folder, 'half') for folder in train]
# cut_normal:用cut2img_reid.py处理normal文件夹后的结果
cam2_file = [os.path.join(directory_path, folder, 'cut_normal') for folder in train]
# print("Cam0 文件路径:", cam0_file)
# print("Cam1 文件路径:", cam1_file)
# print("Cam2 文件路径:", cam2_file)
cam0_files = get_files_in_folders(cam0_file)
cam1_files = get_files_in_folders(cam1_file)
cam2_files = get_files_in_folders(cam2_file)
cam2_file = []

print('c0')
train_list = []
for file in cam0_files:
    id_ = int(os.path.basename(os.path.dirname(os.path.dirname(file))))
    result_tuple = (file, id_, int(0))
    train_list.append(result_tuple)
print('c1')
for file in cam1_files:
    id_ = int(os.path.basename(os.path.dirname(os.path.dirname(file))))
    result_tuple = (file, id_, int(1))
    train_list.append(result_tuple)
print('c2')
for file in cam2_files:
    id_ = int(os.path.basename(os.path.dirname(os.path.dirname(file))))
    result_tuple = (file, id_, int(2))
    train_list.append(result_tuple)
print(len(train_list))

## test_query
print('test_query')
cam0_file = [os.path.join(directory_path, folder, 'normal') for folder in test_query]
cam1_file = [os.path.join(directory_path, folder, 'half') for folder in test_query]
cam2_file = [os.path.join(directory_path, folder, 'cut_normal') for folder in test_query]
print("Cam0 文件路径:", cam0_file)
print("Cam1 文件路径:", cam1_file)
print("Cam2 文件路径:", cam2_file)
cam0_files = get_files_in_folders(cam0_file)
cam1_files = get_files_in_folders(cam1_file)
cam2_files = get_files_in_folders(cam2_file)
cam2_file = []
# print(cam0_files)
# print(cam1_files)
print('split test query')


query_list = []
test_list = []
for file in cam0_files:
    id_ = int(os.path.basename(os.path.dirname(os.path.dirname(file))))
    cam_id = 0
    result_tuple = (file, id_, cam_id)
    if random.random() < 0.3:  # 假设50%的图片为query
        query_list.append(result_tuple)
    else:
        test_list.append(result_tuple)

for file in cam1_files:
    id_ = int(os.path.basename(os.path.dirname(os.path.dirname(file))))
    cam_id = 1
    result_tuple = (file, id_, cam_id)
    if random.random() < Query_Ratio:  # 假设50%的图片为query
        query_list.append(result_tuple)
    else:
        test_list.append(result_tuple)

for file in cam2_files:
    id_ = int(os.path.basename(os.path.dirname(os.path.dirname(file))))
    cam_id = 2
    result_tuple = (file, id_, cam_id)
    if random.random() < Query_Ratio:  # 假设50%的图片为query
        query_list.append(result_tuple)
    else:
        test_list.append(result_tuple)

# 输出列表
# print("Query List:", query_list)
# print("Gallery List:", test_list)

with open(txt_path + '/bounding_box_train.txt', 'w') as file:
    for item in train_list:
        file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

with open(txt_path + '/query.txt', 'w') as file:
    for item in query_list:
        file.write(f"{item[0]}, {item[1]}, {item[2]}\n")


with open(txt_path + '/bounding_box_test.txt', 'w') as file:
    for item in test_list:
        file.write(f"{item[0]}, {item[1]}, {item[2]}\n")