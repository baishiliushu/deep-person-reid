# -*- coding: utf-8 -*-
import os
import shutil
import torch
from torchreid.utils import FeatureExtractor
from tqdm import tqdm  # 用于显示进度条
from PIL import Image

MODEL_TYPE = "osnet_pcb_512d_ibn" #'osnet_x1_0'
MODEL_LOCATION = "/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251105-225225_osnet_pcb_512d_ibn_0.003_id124_pcb6-left-triplet-mark1501/osnet_pcb_512d_ibn-triplet-pre_False_id124_pcb6-left-triplet/model_weights.pth"  #'/home/mount_point_one/ssx/workspace/deep-person-reid/torchreid/models/checkpoints/20241111osnet_x1_0.pth.tar-50'
# Initialize the feature extractor with a pretrained model
extractor = FeatureExtractor(
    model_name=MODEL_TYPE, 
    model_path=MODEL_LOCATION, 
    device='cuda'  # or 'cpu'
)

def extract_feature(image_path):
    features = extractor([image_path])
    return features[0]

def compare_images(image_path1, image_path2):
    similarity = None
    try:
        gallery_img = Image.open(image_path1).convert('RGB')
        current_img = Image.open(image_path2).convert('RGB')
        
        # 进行图片比较的逻辑...
        features1 = extract_feature(image_path1)
        features2 = extract_feature(image_path2)
        similarity = torch.nn.functional.cosine_similarity(features1, features2, dim=0)
        
        return similarity.item()
    except Exception as e:
        print(f"Warning: Failed to load or compare images {image_path1} and {image_path2}: {e}")
        return None
    
    #return similarity.item()

def safe_load_image(img_path):
    """
    安全加载图像，若失败则返回 False。
    
    Args:
        img_path (str): 图像文件路径
    
    Returns:
        PIL.Image.Image or False: 成功返回 RGB 图像，失败返回 False
    """
    if not isinstance(img_path, str) or not os.path.isfile(img_path):
        return False

    try:
        image = Image.open(img_path).convert('RGB')
        # 可选：强制加载像素数据，避免延迟加载导致后续出错
        image.load()
        return image
    except Exception as e:
        # 可选：记录日志
        # print(f"Warning: Failed to load image {img_path}: {e}")
        return False

def add_to_gallary(src_img_dir, filename, gallery_paths, dword, hword):
    gallery_path = os.path.join(src_img_dir, filename)
    g_name = os.path.splitext(filename)[0]
    repeat_folder = os.path.join(src_img_dir, f"{g_name}_{dword}" ) 
    hit_folder = os.path.join(src_img_dir, f"{g_name}_{hword}")   
    gallery_paths.append({"g_path":gallery_path, "r_floder":repeat_folder, "h_floder":hit_folder})
    return gallery_paths

def get_last_part_of_path(path):
    """获取路径的最后一级名称（文件名或文件夹名）"""
    # 先去除路径末尾的分隔符（如果存在）
    path = path.rstrip(os.sep)
    # 获取最后一级
    return os.path.basename(path)

def main(src_img_dir, delete_word, hit_word, threshold=0.97, gallary_t=0.8):
    # Get sorted list of image files in directory A
    image_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Step 2: 创建一个列表字典    
    gallery_paths = []
    
    # Set the initial baseline frame without moving it
    if len(image_files) > 1:
        gallery_paths = add_to_gallary(src_img_dir, image_files[0], gallery_paths, delete_word, hit_word)
        # Step 3: 遍历剩余图片（第2张到最后一张）
        for img_file in tqdm(image_files[1:], desc=f"Processing {os.path.basename(src_img_dir)}"):
            current_path = os.path.join(src_img_dir, img_file)

            # 如果当前图片已经被移走（比如之前处理过），跳过
            if not os.path.exists(current_path):
                continue

            is_duplicate = False
            score = 0.0
            similarity_low = {"s":100.0, "g":"", "r_floder": "", "h_floder":""}
            similarity_high = {"s":-100.0, "g":"", "r_floder": "", "h_floder":""}
            
            # 与 gallery 中每张图比较
            for ref in gallery_paths:
                 # [{"g_path", os.path.join(src_img_dir, filename), "r_floder": delete_folder, "h_floder": normal_folder}, ... , {}]
                ref_path = ref["g_path"]
                
                # 再次确认 ref_path 还在（理论上它不会被移，但保险起见）
                if not os.path.exists(ref_path):
                    continue

                score = compare_images(ref_path, current_path)
                
                if score is None:
                    continue  # 比较失败，跳过这次比较
            
                if score < similarity_low["s"]:
                    similarity_low["s"] = score
                    similarity_low["g"] = ref_path
                    similarity_low["r_floder"] = ref["r_floder"]
                    similarity_low["h_floder"] = ref["h_floder"]
                if score > similarity_high["s"]:
                    similarity_high["s"] = score
                    similarity_high["g"] = ref_path
                    similarity_high["r_floder"] = ref["r_floder"]
                    similarity_high["h_floder"] = ref["h_floder"]
                print(f"<S> {score} between {ref_path} [v.s.] {current_path} ")
            # break  按道理循环时有重复的不用再比较了，但是为了多打印得分所以用最值计算
            score_max = similarity_high["s"]
            ref_max = similarity_high["g"]
            delete_folder = similarity_high["r_floder"]
            normal_folder = similarity_high["h_floder"]
            if score_max > threshold:
                is_duplicate = True 
                
                    
                
            print(f"<S>  among gallary {len(gallery_paths)} [v.s.] {current_path} : lowest -> {similarity_low} ; highest -> {similarity_high}")
            if is_duplicate:
                print(f"dup {img_file}")
                try:
                 # Ensure the delete and normal folders exist 
                    os.makedirs(delete_folder, exist_ok=True)

                    shutil.move(current_path, delete_folder)
                    print(f"[Repeat] {score} {current_path} TO {delete_folder} with g {ref_max}")
                except Exception as e:
                    print(f"[ERROR] Failed to move {current_path}: {e}")
            else:
                # 命中： low_t < score < repeat_t
                if score_max > gallary_t:
                    print(f"hit {img_file}")
                    os.makedirs(normal_folder, exist_ok=True)
                    shutil.move(current_path, normal_folder)
                    print(f"[Hit] {score} > {gallary_t} , {current_path} TO h {normal_folder} with g {ref_max}")
                else:
                    # 不重复 → 加入 gallery（作为新代表）
                    print(f"add {img_file}")
                    gallery_paths = add_to_gallary(src_img_dir, img_file, gallery_paths, delete_word, hit_word)
                    print(f"Add  {score} < {gallary_t}  gallary {len(gallery_paths)} from {current_path} ")
    info_g_comapre = ""
    gallary_results = [] #{"s":1.0, "g":""}
    if len(gallery_paths) > 1:
        length = len(gallery_paths)
        for i in range(length):
            for j in range(i + 1, length):
                g_i_path = gallery_paths[i]["g_path"]
                g_j_path = gallery_paths[j]["g_path"]
                gallary_score = compare_images(g_i_path, g_j_path)
                gallary_results.append({"s": gallary_score, "pair":f"{os.path.splitext(get_last_part_of_path(g_i_path))[0]} v.s. {os.path.splitext(get_last_part_of_path(g_j_path))[0]}"})
         
    print("\n\n[Finished]{} ,    gallary_length {}    -> {}".format(src_img_dir,len(gallery_paths),  gallary_results))


def get_dir_imgs(root_dir):
    """
    遍历 root_dir 下所有子目录，返回包含至少一个 .jpg 或 .jpeg 文件的目录路径列表（排序）。
    
    Args:
        root_dir (str): 根目录路径
    
    Returns:
        List[str]: 排序后的目录路径列表
    """
    jpg_dirs = set()  # 用 set 避免重复

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 检查当前目录是否有 .jpg / .jpeg 文件（不区分大小写）
        has_jpg = any(
            f.lower().endswith(('.jpg', '.jpeg', 'png'))
            for f in filenames
        )
        if has_jpg:
            jpg_dirs.add(dirpath)

    # 转为列表并排序（字典序）
    results = sorted(jpg_dirs)
    return results

from pathlib import Path

def restore_repeat_images(root_dir):
    """
    遍历 root_dir，将所有名为 'repeat' 的子目录中的图片移回其父目录。
    
    Args:
        root_dir (str): 顶层目录路径
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    # 查找所有名为 'repeat' 的目录
    repeat_dirs = list(root.rglob("repeat"))
    
    print(f"Found {len(repeat_dirs)} 'repeat' directories under {root_dir}")
    
    moved_count = 0
    for repeat_dir in repeat_dirs:
        if not repeat_dir.is_dir():
            continue
        
        parent_dir = repeat_dir.parent
        print(f"\nProcessing: {repeat_dir} -> {parent_dir}")
        
        # 遍历 repeat 目录中的所有图片文件
        for item in repeat_dir.iterdir():
            if item.is_file() and item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
                target_path = parent_dir / item.name
                
                # 如果目标已存在，可选择跳过或重命名（这里选择跳过并警告）
                if target_path.exists():
                    print(f"  ⚠️Skip {item.name}: already exists in parent")
                    continue
                
                try:
                    shutil.move(str(item), str(target_path))
                    print(f"  ✅ Moved: {item.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ❌ Failed to move {item}: {e}")
        # 可选：如果 repeat 目录现在为空，可以删除它
        #try:
            #if not any(repeat_dir.iterdir()):
                #repeat_dir.rmdir()
                #print(f"  🗑️  Removed empty 'repeat' directory")
        #except Exception as e:
            #print(f"  ⚠️  Could not remove 'repeat' dir: {e}")
    
    print(f"\n✅ Restoration complete. Total files moved: {moved_count}")



"""
统计量：
rank1-N 的图片覆盖率，rank1-N之间的相似度；
person1-M rank1-N之间的图片覆盖率，rank1-N之间的相似度距离。

# 1.去重main
# 2.手动合并ID（repeat也剪切，影响分母）
# 3-A.遍历ID，阈值0.7下的gallary
# 3-B.遍历ID，rank10时的阈值和不可分项
# 

待考虑条件：先简单0.9去重 -> 导致统计问题，分母、分子都变小（分母变小更多一些）-> 可以剪切过来
split_dir TO [normal, half, useless] ? OR script-first_manual-again -> NO,  manual first
---<dir1_personA_normal>
    |---org-filename_timestamp.jpg    # which is compare({Gn}) < 0.7
    |---org-filename_timestamp.jpg  #... log -> lowest/highest_similarity
    |---...                           #... rename -> add-endless(_g-i)
    |---org-filename_timestamp.jpg    
    |---[org-fa_i_repeat]            # which is compare({Gn})> 0.9
    |---[org-fa_i_hit]               # which is 0.7 < compare({Gn}) < 0.9
---dir1_personA_half
    |---
---dir1_personA_useless
    |---

# repeat不参与对比，除此之外都参与，在没有分人的时候不使能hit目录，因为得不到同一个人ID内全部图片，也就拿不到差异
compare({dir-X:Gallary-imgs})

compare_with_crop()
"""

def compare_one_to_multi(img, gallarys):
    result_dict = {"lowest_cos": 0.0, "lowest_path": "", "highest_socre": 0.0, "highest_path":""}
    
    return result_dict

def main_split_dir(cam_folder, repeat_keyword, hit_keyword, repeat_t=0.97, gallary_t=0.8):        
    dir_rst_dict = {}
    print("[Finished]{}".format(cam_folder))
    return dir_rst_dict


if __name__ == "__main__":
    #base_dir = "/home/leon/mount_point_d/test-result-moved/reid_datas/251104_track_1104_ROI"
    base_dir = "/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/compare_images/1113-r550" #"/home/leon/mount_point_d/test-result-moved/reid_datas/202511_rental_house/id_with_datasets_format/compare-shoes"
#"/home/leon/mount_point_d/test-result-moved/reid_datas/202511_rental_house/id_with_datasets_format/compare"
#"/home/leon/mount_point_d/test-result-moved/reid_datas/202511_rental_house/rois"
    #restore_repeat_images(base_dir)
    #exit(0)
    high_th = 1.0
    low_thesould = 1.0
    repeat_key = "repeat"
    hit_key = "hit"
    las_dir_names = get_dir_imgs(base_dir)
    for d in las_dir_names:
        jump_flag = False
        if repeat_key in d or hit_key in d or "useless" in d:
            jump_flag = True
        if jump_flag:
            print("[JMUP]Already is processd-dir {}".format(d))
            continue
        directory_a = d 
        
        main(directory_a, repeat_key, hit_key, high_th, low_thesould)
        
    print("[DONE] repeat if > {}, gallary if < {}".format(high_th, low_thesould))
    print("[DONE] model -> {}".format(MODEL_TYPE, MODEL_LOCATION))
    
# find . -name "*.jpg" -type f | wc -l
#'/home/indemind/nfs_1/reid_datas/reid_dataset_1104/{}'.format(id_path_name)
#'/home/indemind/nfs_1/reid_datas/reid_dataset_1104/{}/repeat'.format(id_path_name)
#'/home/indemind/nfs_1/reid_datas/reid_dataset_1104/{}/normal_0.85'.format(id_path_name)
# normal_folder = '/home/indemind/nfs_1/reid_datas/reid_dataset_1104/0004/half'

