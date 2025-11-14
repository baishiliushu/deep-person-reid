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
    base_dir = "/home/leon/mount_point_d/test-result-moved/reid_datas/ljw-dataset/data_ids/"
    directory_paths = [
        base_dir + 'our_ids_left',
        base_dir + 'vot19',
        base_dir + 'ut_kinect',
        base_dir + 'otb2015',
        base_dir + 'HIE1',
        base_dir + 'HIE2',
        #base_dir + 'prid_as_ours'
    ]
    output_dir = base_dir + 'data_all_6_124'
    id = 0
    for directory in directory_paths:
        valid_subfolders = get_valid_subfolders(directory, output_dir)
