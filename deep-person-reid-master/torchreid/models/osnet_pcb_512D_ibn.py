from __future__ import division, absolute_import
import torch
from torch import nn
import torch.nn.functional as F
from .osnet import osnet_ibn_x1_0

class OSNetPCB512DIbn(nn.Module):
    """
    在 OSNet 的基础上加入 PCB 结构，并支持 softmax 和 triplet 损失
    推理时始终返回 512 维特征
    """

    def __init__(self, num_classes, parts=4, loss='softmax', pretrained=False, **kwargs):
        super(OSNetPCB512DIbn, self).__init__()
        # 使用 osnet_ibn_x1_0 作为 backbone
        self.backbone = osnet_ibn_x1_0(num_classes=1000, pretrained=pretrained, **kwargs)
        self.parts = parts
        self.num_features = 512
        self.loss = loss

        # 为每个 part 设立独立分类器
        self.classifiers = nn.ModuleList([
            nn.Linear(self.num_features, num_classes) for _ in range(parts)
        ])

    def forward(self, x):
        # 提取特征图，尺寸形状为 [B, C, H, W]
        feat_map = self.backbone.featuremaps(x)
        B, C, H, W = feat_map.size()
        part_h = H // self.parts
        part_features = []

        # 提取每个 part 的特征
        for i in range(self.parts):
            part = feat_map[:, :, i * part_h: (i + 1) * part_h, :]
            part_feat = part.mean([2, 3])  # shape: [B, C]
            part_features.append(part_feat)

        # 推理时始终返回 512 维特征（平均所有 part）
        if not self.training:
            # 堆叠后取平均，得到 512 维特征
            part_features_tensor = torch.stack(part_features, dim=1)  # 形状：[B, num_parts, 512]
            averaged_features = part_features_tensor.mean(dim=1)  # 形状：[B, 512]
            return averaged_features

        # 训练时根据 loss 类型返回不同格式
        logits = [self.classifiers[i](part_feat) for i, part_feat in enumerate(part_features)]

        if self.loss == 'softmax':
            # Softmax 模式：只返回 logits（用于分类损失）
            return logits
        elif self.loss == 'triplet':
            # Triplet 模式：返回 (logits, 512维特征)
            # 使用平均特征作为 triplet loss 的输入
            averaged_features = torch.stack(part_features, dim=1).mean(dim=1)  # [B, 512]
            return logits, averaged_features
        elif self.loss == 'softmax_triplet':
            # 联合损失模式：返回 (logits, 512维特征)
            averaged_features = torch.stack(part_features, dim=1).mean(dim=1)  # [B, 512]
            return logits, averaged_features
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

# 为了和其他模型一致，提供一个接口函数
def osnet_pcb_512D_Ibn(num_classes=1000, loss='softmax', pretrained=True, **kwargs):
    model = OSNetPCB512DIbn(num_classes=num_classes, parts=6, loss=loss, pretrained=pretrained, **kwargs)
    return model




