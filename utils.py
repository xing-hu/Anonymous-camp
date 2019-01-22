from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import os
import random
from torch.autograd import Variable
import torch


class ImageDataset(Dataset):
    def __init__(self, root, A_transforms_=None,B_transforms_=None, mode='train'):
        self.A_transforms = transforms.Compose(A_transforms_)
        self.B_transforms = transforms.Compose(B_transforms_)
        self.files_A = glob.glob(os.path.join(root, mode + '_A') + '/*.*')
        self.files_B = glob.glob(os.path.join(root, mode + '_B') + '/*.*')

    def __getitem__(self, index):
        item_A = self.A_transforms(Image.open(self.files_A[random.randint(0, len(self.files_A) - 1)]))
        item_B = self.B_transforms(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
