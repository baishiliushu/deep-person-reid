'''
python reid.py --image_path1 input/1.jpg --image_path2 input/2.jpg
'''
import numpy as np

import torchreid
import torch
from PIL import Image
from torchreid.utils import FeatureExtractor
import argparse
import os

MODEL_DIR = "/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251105-225225_osnet_pcb_512d_ibn_0.003_id124_pcb6-left-triplet-mark1501/osnet_pcb_512d_ibn-triplet-pre_False_id124_pcb6-left-triplet/model.pth.tar-120" 
#"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251103-175705_osnet_pcb_512d_ibn_0.003_id164_softmax_onlyfilp/osnet_pcb_512d_ibn-softmax-pre_False_id164_softmax_onlyfilp/model/model.pth.tar-120"
#"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251103-144834_osnet_pcb_512d_ibn_0.003_id164_triplet_onlyfilp/osnet_pcb_512d_ibn-triplet-pre_False_id164_triplet_onlyfilp/model_weights.pth"
# "/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251103-184454_osnet_pcb_512d_ibn_0.003_id164_pcb4/osnet_pcb_512d_ibn-softmax-pre_False_id164_pcb4/model_weights.pth"
#"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251103-162422_osnet_pcb_512d_ibn_0.003_id164_pre-softmax_onlyfilp/osnet_pcb_512d_ibn-softmax-pre_True_id164_pre-softmax_onlyfilp/model/model.pth.tar-200"
#"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251103-120712_osnet_pcb_512d_ibn_0.003_id164_based_pre-tpt_onlyfilp/osnet_pcb_512d_ibn-triplet-pre_True_id164_based_pre-tpt_onlyfilp/model/model.pth.tar-200"
#"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251031-174147_osnet_pcb_512d_ibn_0.003/osnet_pcb_512d_ibn-softmax-pre_False_prid-ids164-erz_p/model/model.pth.tar-60"
#"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251031-184310_osnet_pcb_512d_ibn_0.003/osnet_pcb_512d_ibn-softmax-pre_False_prid-ids352-erz_p/model/model.pth.tar-120"
#/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251031-174147_osnet_pcb_512d_ibn_0.003/osnet_pcb_512d_ibn-softmax-pre_False_prid-ids164-erz_p 
# /home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251029-202245_osnet_pcb_521d_ibn/osnet_pcb_521d_ibn-softmax-pre_False_only_ids 
#/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251030-162344_osnet_ibn_x1_0/osnet_ibn_x1_0-softmax-pre_False_id456/model/model.pth.tar-120
# /home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251029-150458_osnet_pcb_521d_ibn/osnet_pcb_521d_ibn-softmax-pre_True_ids456/model/model.pth.tar-120 
# "/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251029-102110/osnet_pcb_521d_ibn-softmax-pre_False_ids456/model/model.pth.tar-60"   /home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251028-201723/osnet_pcb_521d-softmax-pre_False_ids456/model/model.pth.tar-60
# Initialize the feature extractor with a pretrained model
extractor = FeatureExtractor(
    model_name='osnet_pcb_512d_ibn',  # You can choose other models too, like 'resnet50', 'resnet101' osnet_x1_0  osnet_pcb_512d_ibn osnet_pcb_512d  
    model_path=MODEL_DIR, 
    # None to use default pretrained model
    device='cuda'  # or 'cpu'
    # device='cpu'  # or 'cpu'
)
#'/home/spring/test_qzj/project/deep-person-reid-master/log_dyf/20250212_osnet_pcb'
               #'(lr=0.003)/model/model'
               #'.pth'
               #'.tar'
               #'-60',

crop_ratios = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7,0.85]



def resize_if_needed(image_path, target_size=(128, 256)):
    # 打开图像文件
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        # 获取原始尺寸
        original_width, original_height = img.size
        print(f"原始尺寸: 宽={original_width}, 高={original_height}")

        # 如果尺寸与目标尺寸相同，则不进行任何操作
        if (original_width, original_height) == target_size:
            print("图片尺寸合适，无需调整")
            return img
        else:
            print(f"调整图片大小从 {original_width}x{original_height} 到 {target_size[0]}x{target_size[1]}")
            # 调整大小
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)  # 使用高质量的重采样滤波器
            #resized_img.save(image_path)
            return resized_img

def crop_and_resize(image_path):

    #image = Image.open(image_path).convert('RGB')
    image = resize_if_needed(image_path)

    width, height = image.size
    resized_images = []
    for ratio in crop_ratios:
        left = 0
        top = height * ratio
        right = width
        bottom = height
        cropped_image = image.crop((left, top, right, bottom))
        resized_image = cropped_image.resize((width, height))
        resized_images.append(resized_image)
    return resized_images

def extract_feature(image_input):
    # 如果传入的是 PIL Image，则转换为 numpy 数组
    if isinstance(image_input, Image.Image):
        image_input = np.array(image_input)
    features = extractor([image_input])  # 现在传入的是 numpy 数组
    return features[0]


def compare_images(image_path1, image_path2):
    # Extract features from both images
    features1 = extract_feature(image_path1)
    features2 = extract_feature(image_path2)
    # Compute the cosine similarity
    similarity = torch.nn.functional.cosine_similarity(features1, features2, dim=0)
    return similarity.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path1', type=str, default='/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/compare_images/1113-r550/g-12.png')
    parser.add_argument('--image_path2', type=str, default='/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/compare_images/1113-r550/q-17.png')
    parser.add_argument('--full_half', type=int, default='1') #是否为全身照与半身照对比，需要image1为全身、image2为半身
    opt = parser.parse_args()

    image_path1 = opt.image_path1
    
    image_path2 = opt.image_path2

    full_half = opt.full_half
    print(MODEL_DIR)
    gallary_images = []
    dirname = ""
    
    if os.path.isdir(image_path1):
        gallary_images = sorted([f for f in os.listdir(image_path1) if f.endswith(('.jpg', '.png', '.jpeg', '.png'))])
        dirname = image_path1
    else:
        gallary_images.append(os.path.basename(image_path1))
        dirname = os.path.dirname(image_path1)
    for _p in gallary_images:
        img_gallary_path = os.path.join(dirname, _p)
        
        image_path1 = img_gallary_path
        image_path1_crop = crop_and_resize(image_path1)
        image_path2_crop = crop_and_resize(image_path2)
        similarity_score = dict()
        similarity_min = dict()
        similarity_score["ratio"] = -1
        similarity_score["s"] = 0.00
        similarity_min["ratio"] = -1
        similarity_min["s"] = 1.00
        similarity_fused = dict()
        similarity_fused["ratio"] = ""
        similarity_fused["s"] = 1.00

        if full_half == 0:
            similarity_score["s"] = compare_images(image_path1, image_path2)
        if full_half == 1:
            #similarity_score = max([compare_images(cropped_img, image_path2) for cropped_img in image_path1_crop])
            
            _index = 0
            for cropped_img in image_path1_crop:
                _s = compare_images(cropped_img, image_path2)
                print("No.{} zero-top MOVE-DOWN h * ratio ({})'s cos is {}  ".format(_index, crop_ratios[_index], _s))
                if crop_ratios[_index] == 0.85:
                    print("do not consider ratio {} .".format(crop_ratios[_index]))
                else:
                    if _s > similarity_score["s"]:
                        similarity_score["s"] = _s
                        similarity_score["ratio"] = crop_ratios[_index]
                    if _s < similarity_min["s"]:
                        similarity_min["s"] = _s
                        similarity_min["ratio"] = crop_ratios[_index]
                _index = _index + 1
        if full_half > 0:
            # 1. 计算image_path1图像的根据不同剪裁比例构建的全部gallery所融合出的特征
            # 提取image_path1所有剪裁版本的特征（排除0.85比例）
            features_list = []
            _index = 0
            for cropped_img in image_path1_crop:
                if crop_ratios[_index] != 0.85:  # 排除0.85比例
                    feature = extract_feature(cropped_img)
                    features_list.append(feature)
                _index += 1
            
            # 融合特征：使用平均融合
            if features_list:
                fused_feature = torch.stack(features_list, dim=0).mean(dim=0)
            else:
                # 如果没有有效特征，使用全图特征作为fallback
                fused_feature = extract_feature(image_path1)
            
            # 2. 用image_path1的融合特征与image_path2计算相似度并更新至similarity_score
            feature_image2 = extract_feature(image_path2)
            similarity_fused["s"] = torch.nn.functional.cosine_similarity(fused_feature, feature_image2, dim=0).item()
            similarity_fused["ratio"] = "fused"  # 标记为融合特征
            
            # 3. 用image_path1的全图特征与image_path2计算相似度并更新至similarity_min
            #feature_image1_full = extract_feature(image_path1)
            #similarity_full = torch.nn.functional.cosine_similarity(feature_image1_full, feature_image2, dim=0)
            #similarity_min["s"] = similarity_full.item()
            #similarity_min["ratio"] = 0.0  # 全图对应ratio=0.0
        if full_half == 3:
            #TODO 3-1. 计算image_path1的512维特征F1_512
            #TODO 3-2. 计算image_path2的512维特征F2_512
            #TODO 3-3. 降维image_path2的512维特征成256维F2_256，降维方式参考网络训练方式.mean 
            #TODO 3-4. 分别计算相似度：F1_512与F2_512、F1_512的前256维与F2_256、F1_512的后256维与F2_256
            print("NOT SUPPORT.")
        print("Highest of ({}, {} )is {}".format(image_path1.split("/")[-1], image_path2.split("/")[-1], similarity_score))
        print("Lowest of ({}, {}) is {}".format(image_path1.split("/")[-1], image_path2.split("/")[-1], similarity_min))
        print("Fused of ({}, {}) is {}".format(image_path1.split("/")[-1], image_path2.split("/")[-1], similarity_fused))
        threshold = 0.7  # Adjust this threshold based on your use case
        if similarity_score["s"] > threshold:
            print("The images are likely of the same person.")
        else:
            print("The images are likely of different people.")



