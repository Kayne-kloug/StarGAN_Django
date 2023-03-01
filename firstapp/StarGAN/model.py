import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.G_main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.G_main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.D_main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.D_main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

class Final_model(nn.Module):

    def __init__(self, g_conv_dim, c_dim, g_repeat_num, image_size, d_conv_dim, d_repeat_num, g_lr, d_lr, beta1, beta2, lambda_rec, lambda_cls, lambda_gp, device):
        super(Final_model, self).__init__()
        #if self.dataset in ['CelebA', 'RaFD']:
        self.G = Generator(g_conv_dim, c_dim, g_repeat_num)
        self.D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num)
        # elif self.dataset in ['Both']:
        #     self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
        #     self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), g_lr, [beta1, beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), d_lr, [beta1, beta2])
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')

        #self.G.to(device)
        #self.D.to(device)

        self.lambda_rec, self.lambda_cls , self.lambda_gp = lambda_rec, lambda_cls, lambda_gp
        #self.device = device

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def forward(self, x, c_trg, c_org):
        # Discriminator
        if c_org == None:
            real_out_src, real_out_cls = self.D(x)

            #d_loss_real = - torch.mean(real_out_src)
            #d_loss_cls = self.classification_loss(real_out_cls, label_org, 'CelebA')

            fake = self.G(x,c_trg)
            fake_out_src, _ = self.D(fake.detach())

            #d_loss_fake = torch.mean(fake_out_src)

            alpha = torch.rand(x.size(0), 1, 1, 1)
            x_hat = (alpha.cuda() * x + (1 - alpha.cuda()) * fake).requires_grad_(True)

            hat_out_src, _ = self.D(x_hat)

            #d_loss_gp = self.gradient_penalty(hat_out_src, x_hat)
            return real_out_src, real_out_cls, fake_out_src, hat_out_src, x_hat

        #d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp

        # self.reset_grad()
        # d_loss.backward()
        # self.d_optimizer.step()

        # Generator
        else:
            fake = self.G(x, c_trg)
            fake_out_src, fake_out_cls = self.D(fake)

            #g_loss_fake = - torch.mean(out_src)
            #g_loss_cls = self.classification_loss(out_cls, label_trg, 'CelebA')

            x_reconst = self.G(fake, c_org)

            g_loss_rec = torch.mean(torch.abs(x - x_reconst))

            #g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls

            # self.reset_grad()
            # g_loss.backward()
            # self.g_optimizer.step()
            return fake_out_src, fake_out_cls, x_reconst, fake




