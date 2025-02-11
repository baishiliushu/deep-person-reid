'''
python reid.py --image_path1 input/1.jpg --image_path2 input/2.jpg
'''
import torchreid
import torch
from PIL import Image
from torchreid.utils import FeatureExtractor
import argparse

# Initialize the feature extractor with a pretrained model
extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',  # You can choose other models too, like 'resnet50', 'resnet101'
    model_path='/home/spring/test_qzj/project/deep-person-reid-master/log_dyf/20250211_osnet_ain_x1_0('
               'lr=0.003)/model/model.pth'
               '.tar'
               '-60',
    # None to use default pretrained model
    device='cuda'  # or 'cpu'
    # device='cpu'  # or 'cpu'
)


def extract_feature(image_path):
    # 直接传递图像路径给 FeatureExtractor
    features = extractor([image_path])  # Extract features
    return features[0]  # Return the feature vector


def compare_images(image_path1, image_path2):
    # Extract features from both images
    features1 = extract_feature(image_path1)
    features2 = extract_feature(image_path2)

    # Compute the cosine similarity
    similarity = torch.nn.functional.cosine_similarity(features1, features2, dim=0)
    return similarity.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path1', type=str, default='compare_images/b1-1.jpg')
    parser.add_argument('--image_path2', type=str, default='compare_images/b1-2.jpg')
    opt = parser.parse_args()

    image_path1 = opt.image_path1
    image_path2 = opt.image_path2

    similarity_score = compare_images(image_path1, image_path2)

    print(similarity_score)
    threshold = 0.7  # Adjust this threshold based on your use case
    if similarity_score > threshold:
        print("The images are likely of the same person.")
    else:
        print("The images are likely of different people.")