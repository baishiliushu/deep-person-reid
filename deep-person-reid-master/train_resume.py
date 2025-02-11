import time

import torchreid

#输出ONNX
#python export.py  -d --imgsz 256 128 --include onnx -p "/home/spring/test_qzj/project/deep-person-reid-master/log/20241108/osnet_x1_0/model/osnet_x1_0.pth.tar-50"
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
#     'msmt17_2': MSMT17_2
# }
datamanager = torchreid.data.ImageDataManager(
    root="/home/spring/test_qzj/data",
    # sources=["Occluded_REID", "market1501"],
    # targets=["Occluded_REID", "market1501"],
    sources=["market1501", "newdataset"],
    targets=["newdataset"],
    height=256,
    width=128,
    batch_size_train=64,
    batch_size_test=16,  # 16
    transforms=["random_flip"]
)
model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=datamanager.num_train_pids,
    loss="triplet",
    pretrained=False
)
#
model = model.cuda()
#
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
engine = torchreid.engine.ImageTripletEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)
start_epoch = torchreid.utils.resume_from_checkpoint(
    '/home/spring/test_qzj/project/deep-person-reid-master/base/version17_no_test/model/model.pth.tar-60',
    model,
    optimizer
)


engine.run(
    save_dir="base/version17_no_test_1",
    max_epoch=180,
    eval_freq=10,
    start_epoch=start_epoch
)
