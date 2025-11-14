import os

# 要创建的文件夹名称
folders_to_create = ["no_use", "normal", "half", "special"]

def create_folders(path):
    # 检查每个子文件夹是否存在，不存在则创建
    for folder in folders_to_create:
        folder_path = os.path.join(path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"已创建文件夹: {folder_path}")
        else:
            print(f"文件夹已存在，跳过: {folder_path}")

def traverse_and_create(root_dir):
    # 仅遍历根目录下的直接子文件夹
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):  # 只处理目录
            create_folders(dir_path)

# 设置根目录
root_directory = "/home/indemind/nfs_1/reid_datas/reid_dataset_1104"  # 将这里替换为你的目标目录
traverse_and_create(root_directory)
