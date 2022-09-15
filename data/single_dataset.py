import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_paths = self.A_paths[index]

        A_img = Image.open(A_paths).convert('RGB')

        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_paths}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
