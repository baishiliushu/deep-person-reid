from __future__ import division, print_function, absolute_import
import os.path as osp

from ..dataset import ImageDataset

# Log
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred


class MSMT17_2(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = 'msmt17_2'

    # dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = osp.join(
            self.dataset_dir, 'msmt17/bounding_box_train.txt'
        )
        self.query_dir = osp.join(self.dataset_dir, 'msmt17/query.txt')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'msmt17/bounding_box_test.txt'
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=False)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(MSMT17_2, self).__init__(train, query, gallery, **kwargs)

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
