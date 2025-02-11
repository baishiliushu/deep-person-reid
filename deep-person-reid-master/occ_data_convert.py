import os
import random
import re
##输入数据集路径
directory_path = '/home/spring/test_qzj/data/Occluded_REID/body_images/'
txt_path = '/home/spring/test_qzj/data/Occluded_REID/'
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
def count_and_sample_subfolders(directory, sample_ratio=0.2):
    # 获取目录下的所有子文件夹
    folders = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
    # 计算要抽取的文件夹数量
    sample_count = max(1, int(len(folders) * sample_ratio))
    # 随机选择指定比例的文件夹
    sampled_folders = random.sample(folders, sample_count)

    # 训练集和测试集/查询集的划分
    train_folders = [folder for folder in folders if folder not in sampled_folders]
    print(f"总文件夹数: {len(folders)}")
    print(f"随机选择的文件夹({sample_ratio * 100}%): {sampled_folders}")

    return train_folders, sampled_folders


train, test_query = count_and_sample_subfolders(directory_path)
train_list = []
query_list = []
test_list = []


pattern = re.compile(r'(\d{3})_(\d{2})_')

# 处理训练集文件
for folder in train:
    folder_path = os.path.join(directory_path, folder)
    files = get_files_in_folders([folder_path])
    for file_path in files:
        match = pattern.search(os.path.basename(file_path))
        if match:
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            person_id = person_id - 1
            camera_id = camera_id - 1
            result_tuple = (file_path, person_id, camera_id)
            train_list.append(result_tuple)

# 处理测试集和查询集文件
for folder in test_query:
    folder_path = os.path.join(directory_path, folder)
    files = get_files_in_folders([folder_path])
    for file_path in files:
        match = pattern.search(os.path.basename(file_path))
        if match:
            person_id = int(match.group(1))
            camera_id = int(match.group(2))
            person_id = person_id - 1
            camera_id = camera_id - 1
            # 随机划分到测试集或查询集
            result_tuple = (file_path, person_id, camera_id)
            if random.random() > 0.3:
                test_list.append(result_tuple)
            else:
                query_list.append(result_tuple)

# 输出结果
print("训练集:", train_list)
print("查询集:", query_list)
print("测试集:", test_list)

# 将结果保存到文件
with open(txt_path + '/bounding_box_train.txt', 'w') as file:
    for item in train_list:
        file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

with open(txt_path + '/query.txt', 'w') as file:
    for item in query_list:
        file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

with open(txt_path + '/bounding_box_test.txt', 'w') as file:
    for item in test_list:
        file.write(f"{item[0]}, {item[1]}, {item[2]}\n")
