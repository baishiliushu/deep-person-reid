import time
from torchreid.utils import load_pretrained_weights
import torchreid
import os
#输出ONNX


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

output_path = '15'
weights_path = ('/home/spring/test_qzj/project/deep-person-reid-master/base/20250116_se_resnet50/model/model.pth.tar-60')
model = "se_resnet50"
loss_type = "triplet"  # "softmax"

datamanager = torchreid.data.ImageDataManager(
    root="/home/spring/test_qzj/data",
    sources=["newdataset"],
    targets=["newdataset"],
    height=256,
    width=128,
    batch_size_train=64,
    batch_size_test=16,
    transforms=["random_flip", "random_crop"]
)
model = torchreid.models.build_model(
    name=model,
    num_classes=datamanager.num_train_pids,
    loss=loss_type,
    pretrained=False
)

model = model.cuda()


#pretrain  = False时设置预训练模型

load_pretrained_weights(model, weight_path=weights_path)

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
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

engine.run(
    save_dir=os.path.join('./log_test/osnet_x1_0/', output_path),
    visrank=False,
    test_only=True
)
# engine.run(
#     save_dir="log/20241121_0931_test/osnet_x1_0",
#     max_epoch=10,
#     eval_freq=2,
#     print_freq=10,
#     test_only=False
# )
