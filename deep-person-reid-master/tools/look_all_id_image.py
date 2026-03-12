import os
import shutil

# 源目录路径
source_dir = "/home/leon/mount_point_d/test-result-moved/reid_datas/ljw-dataset/data_ids/data_all_7_171"

# 目标目录路径（请替换为实际路径）
destination_dir = "/home/leon/mount_point_d/test-result-moved/reid_datas/ljw-dataset/data_ids/data_all_7_171_view"  # 请修改为您的目标目录

# 确保目标目录存在
os.makedirs(destination_dir, exist_ok=True)

# 遍历源目录的一级子目录
for root, dirs, files in os.walk(source_dir):
    # 只处理一级子目录（即直接在data_all_7_171下的目录，不处理更深层的目录）
    if root == source_dir:
        for dir_name in dirs:
            # 检查该一级子目录下是否有normal子目录
            normal_dir = os.path.join(root, dir_name, "normal")
            if os.path.isdir(normal_dir):
                # 获取normal子目录下的所有图片文件
                image_files = [f for f in os.listdir(normal_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

                if image_files:
                    # 选择第一张图片（任意一张，这里选择第一张）
                    selected_image = image_files[0]

                    # 构建源文件路径
                    source_file = os.path.join(normal_dir, selected_image)

                    # 构建目标文件名：一级子目录_原文件名
                    new_file_name = f"{dir_name}_{selected_image}"

                    # 构建目标文件路径
                    destination_file = os.path.join(destination_dir, new_file_name)

                    # 复制文件
                    shutil.copy2(source_file, destination_file)
                    print(f"已复制: {source_file} -> {destination_file}")
                else:
                    print(f"警告: {normal_dir} 中没有图片文件")