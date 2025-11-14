import os
from PIL import Image

def crop_and_resize(image_path, output_path):
    # 打开图像
    image = Image.open(image_path)

    # 获取原图尺寸
    width, height = image.size

    # 设置裁剪比例为保留下方 70%
    crop_ratio = 0.7
    crop_start_height = int(height * (1 - crop_ratio))  # 从高度的 30% 开始

    # 定义裁剪区域（保留下方 70%）
    crop_area = (0, crop_start_height, width, height)

    # 裁剪图像下方区域
    cropped_image = image.crop(crop_area)

    # 将裁剪的区域重新调整为原图大小，使用抗锯齿
    resized_image = cropped_image.resize((width, height), Image.LANCZOS)

    # 保存调整后的图片
    resized_image.save(output_path)

def crop_and_resize_left_right(org_path, cut_dir, f_name):
    # 打开图像
    image = Image.open(org_path)

    # 获取原图尺寸
    width, height = image.size

    # 设置裁剪比例
    for crop_ratio in [0.5]:
   
        crop_left_end = int(width * crop_ratio)  # 从left开始

        # 定义裁剪区域l,t, r,b
        crop_area_left = (0, 0, crop_left_end, height)
        cropped_image_left = image.crop(crop_area_left)
        
        crop_area_right = (crop_left_end, 0, width, height)
        cropped_image_right = image.crop(crop_area_right)
        
        # 将裁剪的区域重新调整为原图大小，使用抗锯齿
        resized_image_left = cropped_image_left.resize((width, height), Image.LANCZOS)
        resized_image_right = cropped_image_right.resize((width, height), Image.LANCZOS)
        # 保存调整后的图片
        output_path_left = os.path.join(cut_dir, "{}-left-{}.jpg".format(crop_ratio, f_name))
        resized_image_left.save(output_path_left)
        
        output_path_right = os.path.join(cut_dir, "{}-right-{}.jpg".format(crop_ratio, f_name))
        resized_image_right.save(output_path_right)
        

def process_folders(base_dir, left_flag=True):
    # 遍历 A 目录下的所有文件夹
    for root, dirs, files in os.walk(base_dir):
        # 检查 normal 文件夹
        if os.path.basename(root) == 'normal':
            # 定义 cut_normal 文件夹路径，使其位于 normal 的同级目录
            cut_normal_dir = os.path.join(os.path.dirname(root), 'cut_normal')
            os.makedirs(cut_normal_dir, exist_ok=True)
            if left_flag:
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    file_name = str(file_name).replace(".jpg", "")
                    try:
                        crop_and_resize_left_right(file_path, cut_normal_dir, file_name)
                        print(f"Processed {file_path} and saved to left {cut_normal_dir}/{file_name}")
                    except Exception as e:
                            print(f"Failed to process {file_path}: {e}")
            else:
                # 遍历 normal 文件夹中的所有图像文件
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    output_path = os.path.join(cut_normal_dir, file_name)

                    # 对图像进行裁剪并保存到 cut_normal 文件夹
                    try:
                        crop_and_resize(file_path, output_path)
                        print(f"Processed {file_path} and saved to {output_path}")
                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")

base_dir = "/home/leon/mount_point_d/test-result-moved/reid_datas/ljw-dataset/data_ids/our_ids_left/" #'/home/indemind/nfs_1/reid_datas/reid_dataset_1104/'
process_folders(base_dir)
