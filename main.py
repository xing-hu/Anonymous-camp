from utils import *
import argparse
import itertools
import numpy as np
import sys
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import GeneratorResNet
from models import Discriminator, LargePatchDiscriminator
from models import weights_init_normal

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--root', type=str, default='./data/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--img_height', type=int, default=218, help='size of image height')
parser.add_argument('--img_width', type=int, default=178, help='size of image width')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--crop_size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
parser.add_argument('--gpu_id', type=int, default=-1, help='GPU id')
parser.add_argument('--model_name', type=str, default='cycleGAN_white', help='model name')
parser.add_argument('--rotate_degree', type=int, default=0, help='rotate degree')
parser.add_argument('--lambda_id', type=float, default=0.5, help='lambda_id')
parser.add_argument('--large_patch', type=bool, default=False, help='whether use large patch')
opt = parser.parse_args()

# make output dirs
os.makedirs('saved_models/%s' % (opt.model_name), exist_ok=True)
os.makedirs('images/%s' % (opt.model_name), exist_ok=True)

cuda = opt.gpu_id > -1

# Gs and Ds
G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)
if opt.large_patch:
    D_A = LargePatchDiscriminator()
    D_B = LargePatchDiscriminator()
else:
    D_A = Discriminator()
    D_B = Discriminator()

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.model_name, opt.epoch)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.model_name, opt.epoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.model_name, opt.epoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.model_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_id = opt.lambda_id * lambda_cyc

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                               lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
A_transforms_ = [transforms.CenterCrop((178, 178)),
                 transforms.Resize((300, 300)),
                 transforms.RandomCrop((256, 256)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomAffine(degrees=opt.rotate_degree, fillcolor=(255, 255, 255)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

B_transforms_ = [transforms.Resize((360, 360)),
                 transforms.RandomCrop((256, 256)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomAffine(degrees=opt.rotate_degree, fillcolor=(255, 255, 255)),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# Training data loader
dataloader = DataLoader(ImageDataset("./data/", A_transforms_=A_transforms_, B_transforms_=B_transforms_),
                        batch_size=opt.batch_size, shuffle=True, )
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("./data/", A_transforms_=A_transforms_, B_transforms_=B_transforms_, mode='test'),
    batch_size=1)

#  Training
if opt.large_patch:
    patch = (1, 64, 64)
else:
    patch = (1, 16, 16)
    
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # Adversarial ground truths

        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        #  Train Generators

        optimizer_G.zero_grad()
        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + \
                 lambda_cyc * loss_cycle + \
                 lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        #  Train Discriminator A

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        #  Train Discriminator B

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
            (epoch, opt.n_epochs,
             i, len(dataloader),
             loss_D.item(), loss_G.item(),
             loss_GAN.item(), loss_cycle.item(),
             loss_identity.item(), time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            # Sample a picture
            imgs = next(iter(val_dataloader))
            real_A = Variable(imgs['A'].type(Tensor))
            fake_B = G_AB(real_A)
            real_B = Variable(imgs['B'].type(Tensor))
            fake_A = G_BA(real_B)
            img_sample = torch.cat((real_A.data, fake_B.data,
                                    real_B.data, fake_A.data), 0)
            save_image(img_sample, 'images/%s/%s.png' % (opt.model_name, batches_done), nrow=5, normalize=True)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.model_name, epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.model_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.model_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.model_name, epoch))
