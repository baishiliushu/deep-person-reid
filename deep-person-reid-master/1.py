import os
import random
from collections import defaultdict

# 输入文件路径
input_file = '/home/leon/mount_point_d/test-result-moved/reid_datas/ljw-dataset/data_ids/datasets/bounding_box_train.txt'
# 输出文件夹路径
output_folder = '/home/leon/mount_point_d/test-result-moved/reid_datas/ljw-dataset/data_ids/datasets'
os.makedirs(output_folder, exist_ok=True)

# 查询集比例
Query_Ratio = 0.125

# 初始化字典以按 person_id 分组
id_group = defaultdict(list)

# 读取文件并按 person_id 分组
with open(input_file, 'r') as file:
    for line in file:
        # 每行数据解析
        img_path, person_id, camera_id = line.strip().split(', ')
        person_id, camera_id = int(person_id), int(camera_id)
        id_group[person_id].append((img_path, person_id, camera_id))

# 分割查询集和测试集
query_list = []
gallery_list = []

# 设置随机种子以确保结果可重复
random.seed(42)

for person_id, items in id_group.items():
    total_count = len(items)
    query_count = max(1, int(total_count * Query_Ratio))  # 至少选择一张图片
    query_samples = random.sample(items, query_count)

    # 将选出的样本加入查询集，其余加入测试集
    query_list.extend(query_samples)
    gallery_list.extend([item for item in items if item not in query_samples])

# 输出分割结果
print(f"总样本数: {sum(len(v) for v in id_group.values())}")
print(f"Query 集样本数: {len(query_list)}")
print(f"Gallery 集样本数: {len(gallery_list)}")

# 写入文件
query_file_path = os.path.join(output_folder, 'query.txt')
gallery_file_path = os.path.join(output_folder, 'bounding_box_test.txt')

with open(query_file_path, 'w') as query_file:
    for item in query_list:
        query_file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

with open(gallery_file_path, 'w') as gallery_file:
    for item in gallery_list:
        gallery_file.write(f"{item[0]}, {item[1]}, {item[2]}\n")

print(f"Query 和 Gallery 数据已分别保存到:\n{query_file_path}\n{gallery_file_path}")
