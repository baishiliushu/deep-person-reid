from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss

from ..engine import Engine


class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs = self.model(imgs)

        # 如果模型返回的是 PCB + 全局分支的输出，
        # 假设 outputs 为 (local_logits, global_logit)
        if isinstance(outputs, (tuple, list)):
            # 如果返回的第一个元素还是一个列表（局部分支），则进入该分支
            if len(outputs) == 2 and isinstance(outputs[0], (list, tuple)):
                local_logits, global_logit = outputs
                # 分别计算局部与全局的交叉熵 loss
                loss_local = sum([self.criterion(l, pids) for l in local_logits]) / len(local_logits)
                loss_global = self.criterion(global_logit, pids)
                # 总 loss 可简单相加，或者根据经验加权求和
                loss = loss_local + loss_global

                # 对于准确率，可以分别计算局部和全局的准确率，然后取平均
                acc_local = sum([metrics.accuracy(l, pids)[0] for l in local_logits]) / len(local_logits)
                acc_global = metrics.accuracy(global_logit, pids)[0]
                acc = (acc_local + acc_global) / 2.0
            else:
                # 如果只是一个 logits 列表（例如单纯 PCB），按原方法计算
                loss = sum([self.criterion(output, pids) for output in outputs]) / len(outputs)
                acc = sum([metrics.accuracy(output, pids)[0] for output in outputs]) / len(outputs)
        else:
            loss = self.criterion(outputs, pids)
            acc = metrics.accuracy(outputs, pids)[0]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': acc.item() if hasattr(acc, 'item') else acc
        }

        return loss_summary

