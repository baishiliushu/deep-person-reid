import time

import torch

import torchreid
from torchreid.utils import load_pretrained_weights
import os
from datetime import datetime
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
lr_set = 0.003  #0.0015 0.003
pretrain_model = "/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/251105-144120_osnet_pcb_512d_ibn_0.003_id164_pcb4_pre-triplet-1/osnet_pcb_512d_ibn-triplet-pre_True_id164_pcb4_pre-triplet-1/model/model.pth.tar-120"
#"/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251102-180140_osnet_pcb_512d_ibn_0.003-triplet_pre/osnet_pcb_512d_ibn-triplet-pre_False_1501-pre-triplet/model/model.pth.tar-200"
#'/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/output/251031-230328_osnet_pcb_512d_ibn_0.003-softmax_pre/osnet_pcb_512d_ibn-softmax-pre_False_1501-pre/model/model.pth.tar-200'
#'/home/leon/work_c_p_p/githubs/deep-person-reid/deep-person-reid-master/log/2510271614/osnet_pcb_521d_ibn-softmax-pre_False_/model/model.pth.tar-60'
save_path_base = "log"
model_type = "osnet_pcb_512d_ibn"   # osnet_pcb_512d / osnet_pcb_512d_ibn
trian_description = "id124_m1501_p4-left-triplet" # "once"
save_path_base = os.path.join(save_path_base, datetime.now().strftime("%y%m%d-%H%M%S") +"_" + model_type + "_{}".format(lr_set)) + "_{}".format(trian_description)

datamanager = torchreid.data.ImageDataManager(
    root="/home/leon/work_c_p_p/githubs/data_links/reid_datas/ljw-dataset",
    sources=["newdataset","market1501"], # "prid","newdataset"  "market1501"
    targets=["newdataset","market1501"], # "prid","newdataset" "market1501"
    height=256,
    width=128,
    batch_size_train=64,  #64
    batch_size_test=16,  # 16
    transforms=["random_flip"] # 'random_erase'  'random_patch''color_jitter' 
    # workers=0
)
# name="osnet_x1_0",
model = torchreid.models.build_model(
    name=model_type,
    num_classes=datamanager.num_train_pids,
    loss=loss_type,
    pretrained=pretrain
)

model = model.cuda()

if pretrain:
    load_pretrained_weights(model, pretrain_model)

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="amsgrad",
    lr=lr_set,
    # lr=0.0015/ 0.003
    # optim="adam",
    # lr=0.0003
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

run_save_path = os.path.join(save_path_base, "{}-{}-pre_{}_{}".format(model_type, loss_type, pretrain, trian_description))
print(f"log save into {run_save_path}")
if engine is not None:
    engine.run(
        save_dir=run_save_path, #"log/osnet_ain_x1_0_pcb6",
        max_epoch=120,  # 60
        eval_freq=20,  # -1
        print_freq=50,  # 10
        test_only=False
    )

# 保存模型权重
model_save_path = os.path.join(run_save_path, "model_weights.pth") # "log/osnet_ain_x1_0_pcb6/model_weights.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")

# 如果需要保存整个模型（包括其结构）
full_model_save_path = os.path.join(run_save_path, "full_model.pth")  # "log/osnet_ain_x1_0_pcb6/full_model.pth"
torch.save(model, full_model_save_path)
print(f"Full model saved to {full_model_save_path}")


# while ps -p 21811 > /dev/null; do sleep 5; done; date; sleep 120; python train.py; date

