'''
python reid.py --image_path1 input/1.jpg --image_path2 input/2.jpg
'''
import numpy as np

import torchreid
import torch
from PIL import Image
from torchreid.utils import FeatureExtractor
import argparse

# Initialize the feature extractor with a pretrained model
extractor = FeatureExtractor(
    model_name='osnet_pcb',  # You can choose other models too, like 'resnet50', 'resnet101'
    model_path='/home/spring/test_qzj/project/deep-person-reid-master/log_dyf/20250212_osnet_pcb'
               '(lr=0.003)/model/model'
               '.pth'
               '.tar'
               '-60',
    # None to use default pretrained model
    device='cuda'  # or 'cpu'
    # device='cpu'  # or 'cpu'
)


def crop_and_resize(image_path):
    crop_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    image = Image.open(image_path).convert('RGB')
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
    parser.add_argument('--image_path1', type=str, default='compare_images/d1-4.jpg')
    parser.add_argument('--image_path2', type=str, default='compare_images/g1-2.jpg')
    parser.add_argument('--full_half', type=int, default='1') #是否为全身照与半身照对比，需要image1为全身、image2为半身
    opt = parser.parse_args()

    image_path1 = opt.image_path1
    image_path1_crop = crop_and_resize(image_path1)
    image_path2 = opt.image_path2
    image_path2_crop = crop_and_resize(image_path2)
    full_half = opt.full_half

    if full_half == 0:
        similarity_score = compare_images(image_path1, image_path2)
    else:
        similarity_score = max([compare_images(cropped_img, image_path2) for cropped_img in image_path1_crop])

    print(similarity_score)
    threshold = 0.7  # Adjust this threshold based on your use case
    if similarity_score > threshold:
        print("The images are likely of the same person.")
    else:
        print("The images are likely of different people.")