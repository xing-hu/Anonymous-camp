import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch import Tensor
from PIL import Image
import torch

from models import GeneratorResNet

parser = argparse.ArgumentParser()
parser.add_argument('--check_point', type=str, default='saved_models/G_AB_10.pth',
                    help='check point from which load trained model')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--A_file', type=str, default='test.png', help='path of the data')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id')
opt = parser.parse_args()
cuda = opt.gpu_id > -1

# # Load pretrained model G_AB
G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
if cuda:
    G_AB = G_AB.cuda()
G_AB.load_state_dict(torch.load(opt.check_point))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [transforms.Resize((opt.img_height, opt.img_width)),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
img_transformer = transforms.Compose(transforms_)

# Test data
img = img_transformer(Image.open(opt.A_file))
real_A = Variable(img.type(Tensor))
img_sample = G_AB(real_A)
save_image(img_sample, 'images/sampled.png' % (), nrow=5, normalize=True)
