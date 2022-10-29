from turtle import forward
import torch
import torch.nn as nn

#Discriminator and Generator Implimentation from DCGAN paper

class Discriminator(nn.Module):
    def __init__(self, img_channels, disc_features):
        super(Discriminator, self).__init__()

        #input shape = N x img_channels x 64 x 64

        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, disc_features, kernel_size=4, stride=2, padding=2),
            # 32 x 32
            nn.LeakyReLU(0.2),
            # block(in_channels, out_channels, kernel_size, stride, padding)
            self.block(disc_features, disc_features*2, 4,2,1),
            self.block(disc_features*2, disc_features*4, 4,2,1),
            self.block(disc_features*4, disc_features*8, 4,2,1),

            # After all _block img output is 4 x 4(conv2d below makes it 1 x 1)
            nn.Conv2d(disc_features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # Bias is made False to facilitate the Batch Normalzation
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, noice_channels, img_channels, gen_features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input : N x channels_noice x 1 x 1
            self.block(noice_channels, gen_features*16, 4, 1, 0),
            self.block(gen_features*16, gen_features*8, 4, 2, 1),
            self.block(gen_features*8, gen_features*4, 4, 2, 1),
            self.block(gen_features*4, gen_features*2, 4, 2, 1),
            nn.ConvTranspose2d(
                gen_features*2, img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.gen(x)


# weight Initialization (according to DCGAN paper)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# defining test

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noice_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1)
    print("Discriminator Test Passed")
    gen = Generator(noice_dim, in_channels, 8)
    z = torch.randn((N, noice_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
    print("Generator test passed")


test()