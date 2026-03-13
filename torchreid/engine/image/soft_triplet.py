from torchreid.engine import ImageTripletEngine
from torchreid.losses import CrossEntropyLoss
from torchreid.losses import SoftTripletLoss
from torchreid.losses import TripletLoss as HardTripletLoss

class SoftTripletEngine(ImageTripletEngine):
    def __init__(self, datamanager, model, optimizer, scheduler=None,
                 margin=0.3, weight_t=1.0, weight_x=1.0, label_smooth=True):
        super().__init__(datamanager, model, optimizer, scheduler,
                         margin=margin, weight_t=weight_t, weight_x=weight_x,
                         label_smooth=label_smooth)

        # 替换 criterion_t
        if margin == 0.0:
            self.criterion_t = SoftTripletLoss()
        else:
            # 保留原来的 hard-margin triplet loss
            
            self.criterion_t = HardTripletLoss(margin=margin)

        # criterion_x 保持不变
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            label_smooth=label_smooth
        )
