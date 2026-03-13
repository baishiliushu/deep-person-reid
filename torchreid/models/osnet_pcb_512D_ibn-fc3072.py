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

    def __init__(self, num_classes=1000, parts=6, loss='softmax', pretrained=False, **kwargs):
        super(OSNetPCB512DIbn, self).__init__()
        # 使用 osnet_ibn_x1_0 作为 backbone
        self.backbone = osnet_ibn_x1_0(num_classes, pretrained=pretrained, **kwargs)
        self.parts = parts
        self.num_features = 512
        self.loss = loss
        self.net_name = "osnet-pcb-512d-ibn"
        print("[counts FOR netword]part: {}, features: {}".format(self.parts, self.num_features))
        # 为每个 part 设立独立分类器
        self.classifiers = nn.ModuleList([
            nn.Linear(self.num_features, num_classes) for _ in range(parts)
        ])
        # 需要每个part的特征必须独立用于triplet loss ，头和头算、脚和脚算 
        self.avgpools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(parts)])
        self.fc =  self._construct_fc_layer(self.num_features, self.num_features * self.parts, dropout_p=None)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 提取特征图，尺寸形状为 [B, C, H, W]
        feat_map = self.backbone.featuremaps(x)
        B, C, H, W = feat_map.size()
        part_h = H // self.parts
        part_features = []
        averaged_features = None

        # 提取每个 part 的特征
        for i in range(self.parts):
            part = feat_map[:, :, i * part_h: (i + 1) * part_h, :]
            part_features.append(part)

        # 512*6 dims 加全连接层
        if self.fc is not None:
            for i in range(self.parts):
                part_features[i] = self.global_avgpool(part_features[i])
                part_features[i] = part_features[i].view(part_features[i].size(0), -1)
            all_feature = torch.cat(part_features, dim=1)
            averaged_features = self.fc(all_feature)
        else:
            for i in range(self.parts):
                part_features[i] = part_features[i].mean([2, 3])  # shape: [B, C]
            # 堆叠后取平均，得到 512 维特征
            part_features_tensor = torch.stack(part_features, dim=1)  # 形状：[B, num_parts, 512]
            averaged_features = part_features_tensor.mean(dim=1)  # 形状：[B, 512] # 推理时始终返回 512 维特征（平均所有 part）

        if not self.training:
            # averaged_features = F.normalize(averaged_features, p=2, dim=1) # 看量化表现是否要归一化  F.normalize
            return averaged_features

        # 训练时根据 loss 类型返回不同格式
        logits = [self.classifiers[i](part_feat) for i, part_feat in enumerate(part_features)]
        
        if self.loss == 'softmax':
            # Softmax 模式：只返回 logits（用于分类损失） 
            return logits
        elif self.loss == 'triplet':    
            # triplet 是否要用cosine计算
            return logits, averaged_features
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
    
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            print("[Debug] fc_layer return None.")
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        print("[Debug]enable in full-connection layers {}".format(layers))
        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)    

# 为了和其他模型一致，提供一个接口函数
def osnet_pcb_512D_Ibn(num_classes=1000, loss='softmax', pretrained=True, **kwargs):
    model = OSNetPCB512DIbn(num_classes=num_classes, parts=6, loss=loss, pretrained=pretrained, **kwargs)
    return model

"""
print("[Debug]net type {}".format(self.model._get_name))
# 训练循环（仅需修改1行）
_, part_features = model(anchor)  # part_features = [feat0, feat1, ...]
triplet_loss = sum(
    triplet_criterion(
        F.normalize(part_features[i], p=2, dim=1),
        F.normalize(positive_features[i], p=2, dim=1),
        F.normalize(negative_features[i], p=2, dim=1)
    ) for i in range(len(part_features))
) / len(part_features)

feature_output = None
            for i, x in enumerate(part_features): 
                v = self.avgpools[i](x) 
                v = v.view(v.size(0), -1)
                #v = self.fcs[i](v)
                avgs.append(v)
            # 选项1：使用所有部分的平均特征（推荐）
            # averaged_features = torch.mean(torch.stack(avgs), dim=0)

            # 选项2：使用所有部分的特征拼接（如果需要保留多部分信息）
            # concatenated_features = torch.cat(avgs, dim=1)
            
            # 选项3：只使用最后一个部分的特征
            # concatenated_features = avgs[-1]
            
            
            #feature_output = torch.cat(avgs, dim=1)
#averaged_features = torch.stack(part_features, dim=1).mean(dim=1)  # [B, 512]
            #------            
            #v = self.global_avgpool(x)
            #v = v.view(v.size(0), -1)
            #if self.fc is not None:
                #v = self.fc(v)
            #------ 
            #concat_features = torch.cat(part_features, dim=1)  # [B, 3072]   

---
        elif self.loss == 'triplet':    
            
            avgs = []
            for i in range(self.parts):
                #v = part_features_org[i]
                
                v = self.avgpools[i](part_features_org[i])  # 对特征图[batch, channels, H, W]应用池化
                v = F.normalize(v, p=2, dim=1)
                v = v.view(v.size(0), -1)
                #[疑问]需不需要计算fc v = self.fc[i](v)
                avgs.append(v)
            
            return logits, avgs 
----
elif self.loss == 'triplet':    
            avgs = []
            for i in range(self.parts):
                #v = part_features_org[i]
                
                v = self.avgpools[i](part_features_org[i])  # 对特征图[batch, channels, H, W]应用池化
                #[疑问]需不需要归一化 v = F.normalize(v, p=2, dim=1)
                v = v.view(v.size(0), -1)
                #[疑问]需不需要计算fc v = self.fc[i](v)
                avgs.append(v)
            #[疑问]需不需平均 averaged_features = torch.mean(torch.stack(avgs), dim=1)
            return logits, avgs
"""


