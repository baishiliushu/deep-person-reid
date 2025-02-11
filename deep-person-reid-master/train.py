import time

import torch

import torchreid
from torchreid.utils import load_pretrained_weights
# __image_datasets = {
#     'market1501': Market1501,
#     'cuhk03': CUHK03,
#     'dukemtmcreid': DukeMTMCreID,
#     'msmt17': MSMT17,
#     'viper': VIPeR,
#     'grid': GRID,
#     'cuhk01': CUHK01,
#     'ilids': iLIDS,
#     'sensereid': SenseReID,
#     'prid': PRID,
#     'cuhk02': CUHK02,
#     'university1652': University1652,
#     'cuhksysu': CUHKSYSU,
#     'newdataset': NewDataset,
#     'market1501_2': Market1501_2,
#     'dukemtmcreid_2': DukeMTMCreID_2,
#     'msmt17_2': MSMT17_2,
#     'Occluded_REID':Occluded_REID
# }
##取消imagenet1k预训练权重
loss_type = "triplet"  # "softmax" or "triplet"
pretrain = False
pretrain_model = '/home/spring/test_qzj/project/deep-person-reid-master/base/20241130_baseline_market1501_180/osnet_x1_0/model/model.pth.tar-60'
datamanager = torchreid.data.ImageDataManager(
    root="/home/spring/test_qzj/data",
    sources=["market1501", "newdataset"],
    targets=["market1501","newdataset"],
    height=256,
    width=128,
    batch_size_train=64,
    batch_size_test=16,  # 16
    transforms=["random_flip"]
)
# name="osnet_x1_0",
model = torchreid.models.build_model(
    name="osnet_ain_x1_0",
    num_classes=datamanager.num_train_pids,
    loss=loss_type,
    pretrained=pretrain
)

model = model.cuda()

if pretrain:
    load_pretrained_weights(model, pretrain_model)

optimizer = torchreid.optim.build_optimizer(
    model,
    # optim="adam",
    # lr=0.0003
    optim="amsgrad",
    # lr=0.0015,
    lr=0.005
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=20
)
engine = None
if loss_type == "softmax":
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )
else:
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

if engine is not None:
    engine.run(
        save_dir="log_dyf/20250211_osnet_ain_x1_0(lr=0.003)",
        max_epoch=60,  # 60
        eval_freq=20,  # -1
        print_freq=50,  # 10
        test_only=False
    )

# 保存模型权重
model_save_path = "log_dyf/20250211_osnet_ain_x1_0(lr=0.003)/model_weights.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")

# 如果需要保存整个模型（包括其结构）
full_model_save_path = "log_dyf/20250211_osnet_ain_x1_0(lr=0.003)/full_model.pth"
torch.save(model, full_model_save_path)
print(f"Full model saved to {full_model_save_path}")