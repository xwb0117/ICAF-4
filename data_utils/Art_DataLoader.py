# *_*coding:utf-8 *_*
import os
import warnings
import numpy as np
from torch.utils.data import Dataset
import torch
import h5py
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class ArtImageDataset(Dataset):
    def __init__(self,root = '/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/Data/ArtImage', npoints=2500, split='trainval', class_choice=None, normal_channel=False, cat=None):
        self.cat = cat
        self.npoints = npoints
        self.root = os.path.join(root,f'{cat}/train/annotations')
        self.normal_channel = normal_channel

        self.meta = {}

        train_ids = set()
        train_txt = os.path.join(root,f'{cat}/train.txt')
        with open(train_txt, 'r') as file:
            lines = file.readlines()

        ids = set()

        for line in lines:
            id = line.split(".")[0]
            if id in ids:
                continue
            train_ids.add(id)

        test_ids = set()
        val_ids = set()
        test_txt = os.path.join(root,f'{cat}/test.txt')
        with open(test_txt, 'r') as file:
            lines = file.readlines()

        for line in lines:
            id = line.split(".")[0]
            test_ids.add(id)
            val_ids.add(id)

        if split == 'trainval':
            fns = sorted(os.listdir(self.root))
            fns = [fn for fn in fns if ((fn[0:-5] in train_ids) or (fn[0:-5] in val_ids))]
        elif split == 'train':
            fns = sorted(os.listdir(self.root))
            fns = [fn for fn in fns if ((fn[0:-5] in train_ids))]
        elif split == 'val':
            fns = sorted(os.listdir(self.root))
            fns = [fn for fn in fns if ((fn[0:-5] in val_ids))]
        elif split == 'test':
            fns = sorted(os.listdir(self.root))
            fns = [fn for fn in fns if ((fn[0:-5] in test_ids))]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        self.datapath = []
        # print(os.path.basename(fns))
        for fn in fns:
            fn = os.path.join(self.root,fn)
            self.datapath.append(fn)

        self.classes = {f'{cat}':0}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        fn = self.datapath[index]
        file_name = os.path.basename(fn)
        id = os.path.splitext(file_name)[0]

        h5py_name = os.path.join(f'/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/xuwenbo/ICAF_data/ArtImage_grasp/{self.cat}_new', f'{id}.h5')

        with h5py.File(h5py_name, 'r') as h5_file:
            data = {key: h5_file[key] for key in h5_file.keys()}
            # data = DataGen(fn, self.cat)
            input_points = data['camcs_per_point'][:]
            if not self.normal_channel:
                point_set = input_points[:, 0:3]
            else:
                point_set = input_points[:, 0:6]
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            choice = np.random.choice(len(input_points), self.npoints, replace=True)
            # resample
            point_set = point_set[choice, :]
            point_set = torch.tensor(point_set, dtype=torch.float32)
            # Get all other items
            gt_dict = {}
            try:
                for k, v in data.items():
                    if k == "camcs_per_point":
                        continue
                    elif "per_point" in k:
                        gt_dict[k] = torch.tensor(v[:], dtype=torch.float32)[choice]
                    else:
                        gt_dict[k] = np.array(v)
            except:
                print(fn)

            return point_set, gt_dict, id

    def __len__(self):
        return len(self.datapath)