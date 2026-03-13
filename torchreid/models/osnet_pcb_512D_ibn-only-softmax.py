from __future__ import division, absolute_import
import torch
from torch import nn
import torch.nn.functional as F
from .osnet import osnet_ibn_x1_0  
#from .osnet import osnet_x1_0

class OSNetPCB512DIbn(nn.Module):
    """
    在 OSNet 的基础上加入 PCB 结构：
      - 将特征图在高度方向分为若干部分（parts）
      - 每个部分分别做全局池化并接入独立分类器
    """

    def __init__(self, num_classes, parts=6, pretrained=False, **kwargs):
        super(OSNetPCB512DIbn, self).__init__()
        # 这里以 osnet_ibn_x1_0 为例作为 backbone，
        # 你也可以根据需要选择其他版本 osnet_x1_0 osnet_ibn_x1_0
        self.backbone = osnet_ibn_x1_0(num_classes=1000, pretrained=pretrained, **kwargs)
        self.parts = parts
        # 注意：这里假定 backbone 的最后输出通道数为 512，与 osnet_ibn_x1_0 定义一致
        self.num_features = 512

        # 不再使用原来的全局平均池化和 fc 层，改为 PCB 的分块池化
        # 为每个 part 设立独立分类器
        self.classifiers = nn.ModuleList([
            nn.Linear(self.num_features, num_classes) for _ in range(parts)
        ])

    def forward(self, x):
        # 提取特征图，尺寸形状为 [B, C, H, W]
        feat_map = self.backbone.featuremaps(x)
        B, C, H, W = feat_map.size()
        # 将高度方向均分为 parts 份
        part_h = H // self.parts
        part_features = []
      

        if not self.training:
            for i in range(self.parts):
                # 对每个部分进行平均池化（也可以根据需要使用 AdaptiveAvgPool2d）
                part = feat_map[:, :, i * part_h: (i + 1) * part_h, :]
                # 此处采用简单的全局平均（跨部分区域的空间维度求平均）
                part_feat = part.mean([2, 3])  # shape: [B, C]
                part_features.append(part_feat)
            
            # 堆叠后分组降维
            part_features_tensor = torch.stack(part_features, dim=1)  # 形状：[B, num_parts, 512]
            averaged_features = part_features_tensor.mean(dim=1)  # 形状：[B, 512]
            return averaged_features
        else:
            logits = []
            for i in range(self.parts):
                # 对每个部分进行平均池化（也可以根据需要使用 AdaptiveAvgPool2d）
                part = feat_map[:, :, i * part_h: (i + 1) * part_h, :]
                # 此处采用简单的全局平均（跨部分区域的空间维度求平均）
                part_feat = part.mean([2, 3])  # shape: [B, C]
                part_features.append(part_feat)
                logits.append(self.classifiers[i](part_feat))
            return logits  # 训练时返回各部分的 logits
        
        # if not self.training:
        #     # 推理时可以将各部分特征拼接成一个长特征向量
        #     return torch.cat(part_features, dim=1)
        # else:
        #     # 训练时返回加权后的 logits
        #     # 设置权重系数
        #     weights = [1.0, 1.2, 1.2, 1.2, 1.2, 1.0]  # 中间三个部分的权重为 1.2
        #     weighted_logits = [logit * weight for logit, weight in zip(logits, weights)]
        #     return weighted_logits  # 返回各部分的加权 logits


# 为了和其他模型一致，提供一个接口函数
def osnet_pcb_512D_Ibn(num_classes=1000, pretrained=True, **kwargs):
    model = OSNetPCB512DIbn(num_classes=num_classes, parts=6, pretrained=pretrained, **kwargs)
    return model
