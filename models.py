import torch.nn as nn
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class GeneratorResNet(nn.Module):
    def __init__(self, res_blocks=9):

        class ResidualBlock(nn.Module):
            def __init__(self, in_features):
                super(ResidualBlock, self).__init__()
                # (in_features, H, W)
                conv_block = [nn.ReflectionPad2d(1),
                              nn.Conv2d(in_features, in_features, 3),
                              nn.InstanceNorm2d(in_features),
                              nn.ReLU(inplace=True),
                              nn.ReflectionPad2d(1),
                              nn.Conv2d(in_features, in_features, 3),
                              nn.InstanceNorm2d(in_features)]
                # (in_features, H, W)
                self.conv_block = nn.Sequential(*conv_block)

            def forward(self, x):
                return x + self.conv_block(x)

        super(GeneratorResNet, self).__init__()
        # input (3, 256, 256)
        # Initial convolution block
        model = [nn.ReflectionPad2d(3),  # (3, 262, 262)
                 nn.Conv2d(3, 64, 7),  # (64, 256, 256)
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        # (128, 128, 128 )
        # (256, 64, 64 )

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]
        # (256, 64, 64)
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        # (128, 128, 128)
        # (64, 256, 256)
        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, 7),
                  nn.Tanh()]
        # (3, 256, 256)
        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            # (in_features, H, W)
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # (out_features, H/2, W/2)
            return layers

        # input (3, 256, 256)
        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),  # (64, 128, 128)
            *discriminator_block(64, 128),  # (128, 64, 64)
            *discriminator_block(128, 256),  # (256, 32, 32)
            *discriminator_block(256, 512),  # (512, 16, 16)
            nn.ZeroPad2d((1, 0, 1, 0)),  # (512, 17, 17)
            nn.Conv2d(512, 1, 4, padding=1)  # (1, 16, 16)
        )

    def forward(self, img):
        return self.model(img)


class LargePatchDiscriminator(nn.Module):
    def __init__(self):
        super(LargePatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            # (in_features, H, W)
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # (out_features, H/2, W/2)
            return layers

        # input (3, 256, 256)
        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),  # (64, 128, 128)
            *discriminator_block(64, 128),  # (128, 64, 64)
            nn.ZeroPad2d((1, 0, 1, 0)),  # (128, 65, 65)
            nn.Conv2d(128, 1, 4, padding=1)  # (1, 64, 64)
        )

    def forward(self, img):
        return self.model(img)


if __name__ == '__main__':
    dummy_input = torch.randn((1, 3, 256, 256))
    G = GeneratorResNet()
    D = Discriminator()

    torch.onnx.export(G, dummy_input, "G.onnx", verbose=True)
    torch.onnx.export(D, dummy_input, "D.onnx", verbose=True)
