import os
import shutil

def get_valid_subfolders(directory, output_dir):
    global id
    os.makedirs(output_dir, exist_ok=True)
    valid_subfolders = []
    if os.path.isdir(directory):
        subfolders = [os.path.join(directory, folder) for folder in os.listdir(directory)
                      if os.path.isdir(os.path.join(directory, folder))]
        print('name:', directory.split('/')[-1], '---', len(subfolders), 'ids')
        for subfolder in subfolders:
            subfolder_name = os.path.basename(subfolder)
            # 尝试将名称转换为整数并检查是否为4位
            try:
                number = int(subfolder_name)
                if 0 <= number <= 9999 and len(subfolder_name) == 4:
                    valid_subfolders.append(subfolder)
                    id = id + 1
                    # print(subfolder, id)
                    shutil.copytree(subfolder, os.path.join(output_dir, f"{(id):04d}"))
            except ValueError:
                pass  # 如果无法转换为数字，跳过该文件夹

    else:
        print(f"Warning: {directory} is not a valid directory")
    valid_subfolders.sort(key=lambda x: int(os.path.basename(x)))  # 使用数字进行排序
    return valid_subfolders

if __name__ == '__main__':
    directory_paths = [
        '/home/spring/test_qzj/data/data_ids/our_ids',
        '/home/spring/test_qzj/data/data_ids/vot19',
        '/home/spring/test_qzj/data/data_ids/ut_kinect',
        '/home/spring/test_qzj/data/data_ids/otb2015',
        '/home/spring/test_qzj/data/data_ids/HIE1',
        '/home/spring/test_qzj/data/data_ids/HIE2',
        '/home/spring/test_qzj/data/data_ids/our_ids_20250103'
    ]
    output_dir = '/home/spring/test_qzj/data/data_ids/data_all'
    id = 0
    for directory in directory_paths:
        valid_subfolders = get_valid_subfolders(directory, output_dir)
