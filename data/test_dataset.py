import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
import pickle
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.dir_A = os.path.join('/mnt/data/czy/haimiandou/data', "testA")
        # self.dir_A = os.path.join('/mnt/data/czy/haimiandou/data', "cvA")
        self.A_paths = sorted(make_dataset(self.dir_A))

        # self.pkl_file = os.path.join(opt.dataroot, "crops.pkl")
        # self.heatmaps_dir = os.path.join(opt.dataroot, "heatmaps/real")
        # self.scans_dir = os.path.join(opt.dataroot, "scans_processed")
        # self.samples = pickle.load(open(self.pkl_file, 'rb'))
        # self.samples = [ j for i in self.samples for j in i]

        # random.shuffle(self.samples)

        self.A = {}

    def __getitem__(self, index):
        # returns samples of dimension [channels, z, x, y]
        # convert to torch tensors with dimension [channel, z, x, y]
        A_paths = self.A_paths[index]
        # print('[*]A_path : ',A_path)
        # print('[*]B_path : ',B_path)

        A = np.load(A_paths)
        # if (A==B).all():
        #     print('[!]data error')
        A[A > 800] = 800
        A[A < -200] = -200
        A = (A + 200) / 1000
        A = torch.from_numpy(A[None, :])

        return {
            'A': A,
            'A_paths': A_paths
        }

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'TestDataset'


if __name__ == '__main__':
    # test
    n = TestDataset()
    n.initialize("./data")
    print(len(n))
    print(n[65])
    print(n[66]['A'].size())
