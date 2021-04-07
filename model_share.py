import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator_conv(nn.Module):
    """Fully convolutional Generator network if latent are cubic."""
    def __init__(self, nc=3, conv_dim=64, repeat_num=2):
        super(Generator_conv, self).__init__()
        '''
        encoder
        '''
        self.start_layers = []
        # self.start_layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.Conv2d(nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.start_layers.append(nn.ReLU(inplace=True))
        self.start_part = nn.Sequential(*self.start_layers)

        # Down-sampling layers.
        self.down_layers = []
        curr_dim = conv_dim
        for i in range(2):
            self.down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.down_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            self.down_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.down_part = nn.Sequential(*self.down_layers)
        self.eli_pose_part = nn.Sequential(*self.start_layers, *self.down_layers)

        # Bottleneck layers.
        self.bottle_encoder_layers = []
        for i in range(repeat_num):
            self.bottle_encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_encoder_part = nn.Sequential(*self.bottle_encoder_layers)

        self.encoder = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers)
        '''
        decoder
        '''
        self.bottle_decoder_layers = []
        for i in range(repeat_num):
            self.bottle_decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_decoder_part = nn.Sequential(*self.bottle_decoder_layers)

        # Up-sampling layers.
        self.up_layers = []
        for i in range(2):
            if i ==0:
                self.up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            else:
                self.up_layers.append(
                    nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))

            self.up_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            self.up_layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.up_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        self.up_layers.append(nn.Tanh())
        self.up_part = nn.Sequential(*self.up_layers)

        self.decoder = nn.Sequential(*self.bottle_decoder_layers, *self.up_layers)

        self.main = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers, *self.bottle_decoder_layers, *self.up_layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or
        x1 = self.encoder(x)
        x2 = self.decoder(x1)

        return x2, x1



    def forward_origin(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)
class Generator_fc(nn.Module):
    """Generator network, with fully connected layers to get latent Z"""
    def __init__(self, nc=3, conv_dim=64, repeat_num=2, z_dim=500):
        self.z_dim = z_dim
        super(Generator_fc, self).__init__()
        '''
        encoder
        '''
        self.start_layers = []
        # self.start_layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.Conv2d(nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.start_layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        self.start_layers.append(nn.ReLU(inplace=True))
        self.start_part = nn.Sequential(*self.start_layers)

        # Down-sampling layers.
        self.down_layers = []
        curr_dim = conv_dim
        for i in range(4):
            if i <= 1:
                self.down_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
                self.down_layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
                self.down_layers.append(nn.ReLU(inplace=True))
                curr_dim = curr_dim * 2
            else:
                self.down_layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
                self.down_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
                self.down_layers.append(nn.ReLU(inplace=True))

        self.down_part = nn.Sequential(*self.down_layers)
        self.eli_pose_part = nn.Sequential(*self.start_layers, *self.down_layers)

        # Encoder Bottleneck layers.
        self.bottle_encoder_layers = []
        for i in range(repeat_num):
            self.bottle_encoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_encoder_part = nn.Sequential(*self.bottle_encoder_layers)

        self.encoder = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers)
        # fc layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Linear(4096, self.z_dim)
        )
        '''
        decoder
        '''
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 256 * 8 * 8)
        )
        # Decoder Bottleneck layers.
        self.bottle_decoder_layers = []
        for i in range(repeat_num):
            self.bottle_decoder_layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.bottlen_decoder_part = nn.Sequential(*self.bottle_decoder_layers)

        # Up-sampling layers.
        self.up_layers = []
        for i in range(4):
            if i <= 1:
                self.up_layers.append(nn.ConvTranspose2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
                self.up_layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
                self.up_layers.append(nn.ReLU(inplace=True))
            else:
                self.up_layers.append(
                    nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
                self.up_layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
                self.up_layers.append(nn.ReLU(inplace=True))
                curr_dim = curr_dim // 2


        self.up_layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        self.up_layers.append(nn.Tanh())
        self.up_part = nn.Sequential(*self.up_layers)

        self.decoder = nn.Sequential(*self.bottle_decoder_layers, *self.up_layers)

        self.main = nn.Sequential(*self.start_layers, *self.down_layers, *self.bottle_encoder_layers, *self.bottle_decoder_layers, *self.up_layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or
        x1 = self.encoder(x)
        x1 = x1.view(x.shape[0], -1)
        z= self.fc_encoder(x1)
        x2 = self.fc_decoder(z)
        x2 = x2.view(x.shape[0], 256, 8, 8)
        x3 = self.decoder(x2)

        return x3, z



    def forward_origin(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator_multi(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_multi, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src
class Discriminator_multi_origin(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_multi, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src

class Discriminator(nn.Module):
        """Discriminator network with PatchGAN."""

        def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
            super(Discriminator, self).__init__()
            layers = []
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))

            curr_dim = conv_dim
            for i in range(1, repeat_num):
                layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
                layers.append(nn.LeakyReLU(0.01))
                curr_dim = curr_dim * 2

            kernel_size = int(image_size / np.power(2, repeat_num))
            self.main = nn.Sequential(*layers)
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

        def forward(self, x):
            h = self.main(x)
            out_src = self.conv1(h)
            return out_src
            # out_cls = self.conv2(h)
            # return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class Discriminator_pose(nn.Module):
    """For pose info elimination, Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, imput_dim=256):
        super(Discriminator_pose, self).__init__()
        layers = []
        conv_dim = conv_dim*2*2
        #　eliminate the last 6 dim to store the pose information
        # layers.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Conv2d(imput_dim, conv_dim*2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim*2
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        # kernel_size = 2
        self.main = nn.Sequential(*layers)
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        # out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_cls.view(out_cls.size(0), out_cls.size(1))
class Discriminator_pose_softmax(nn.Module):
    """For pose info elimination, Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator_pose, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        # kernel_size = int(image_size / np.power(2, repeat_num))
        # kernel_size = 2
        # add Fc layers and use softmax()
        layers.append(nn.Linear(np.square(image_size//(2 * (repeat_num + 1))) * curr_dim, 120))
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.Linear(120, c_dim))
        layers.append(nn.LeakyReLU(0.01))
        self.main = nn.Sequential(*layers)
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        # out_src = self.conv1(h)
        # out_cls = self.conv2(h)
        # return out_cls.view(out_cls.size(0), out_cls.size(1))
        return F.softmax(h)
