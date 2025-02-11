from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import os
from ..dataset import ImageDataset
import re
import glob
import os.path as osp
import warnings

class NewDataset(ImageDataset):
    ##数据集格式/root/new_dataset/datasets

    dataset_dir = 'new_dataset'

    def __init__(self, root='', **kwargs):
        # self.root
        # 将根目录路径转为绝对路径，并支持用户主目录的快捷路径表示（~）。
        # self.dataset_dir
        # 是数据集的完整路径。
        # 数据集文件结构检查
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'datasets')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"datasets".'
            )
        # data_dir ->/root/new_dataset/datasets

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        # train = ...
        # query = ...
        # gallery = ...
        self.train_dir = osp.join(self.data_dir, 'bounding_box_train.txt')
        self.query_dir = osp.join(self.data_dir, 'query.txt')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test.txt')
        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)
        # 处理训练、查询和测试数据集
        # process_dir：加载和处理指定目录下的图片数据。
        # relabel = True：在训练集时，将行人ID重新编码，以便模型识别。
        train = self.process_data_txt(self.train_dir, relabel=True)
        query = self.process_data_txt(self.query_dir, relabel=False)
        gallery = self.process_data_txt(self.gallery_dir, relabel=False)

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

    def process_data_txt(self, dir_path, relabel=False):
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