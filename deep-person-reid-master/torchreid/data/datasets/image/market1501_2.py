from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class Market1501_2(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501_2'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train.txt')
        self.query_dir = osp.join(self.data_dir, 'query.txt')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test.txt')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir)

        super(Market1501_2, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        data_read = []
        pid_container = set()

        # 读取并解析文件内容
        with open(dir_path, 'r') as file:
            for line in file:
                img_path, pid, camid = line.strip().split(', ')
                pid, camid = int(pid), int(camid)
                data_read.append((img_path, pid, camid))
                pid_container.add(pid)
        # 重新编码PID
        if relabel:
            pid2label = {pid: idx for idx, pid in enumerate(pid_container)}
            data_read = [(img_path, pid2label[pid], camid) for img_path, pid, camid in data_read]
        return data_read
