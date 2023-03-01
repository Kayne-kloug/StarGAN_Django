
from .model import Final_model
from .model import Generator
from .model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self):
        """Initialize configurations."""


        # Model configurations.
        self.c_dim = 5
        self.c2_dim = 8
        self.image_size = 256
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.g_repeat_num = 6
        self.d_repeat_num = 6
        self.lambda_rec = 10
        self.lambda_cls = 1
        self.lambda_gp = 10


        # Training configurations.
        self.dataset = "CelebA"
        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]

        # Test configurations.
        self.test_iters = 200000

        # Miscellaneous.

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.

        self.model_save_dir = "firstapp/StarGAN/models"

        # Build the model and tensorboard.
        self.build_model()


    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        #self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        self.Fmodel = Final_model(self.g_conv_dim, self.c_dim, self.g_repeat_num, self.image_size, self.d_conv_dim, self.d_repeat_num, self.g_lr, self.d_lr, self.beta1, self.beta2, self.lambda_rec, self.lambda_cls, self.lambda_gp, self.device)

        self.Fmodel.to(self.device)

        #self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')

        self.G.to(self.device)
        # self.D.to(self.device)

        self.print_network(self.Fmodel, 'Final')

    def print_network(self, Fmodel, name):
        """Print out the network information."""
        num_params = 0
        for p in Fmodel.parameters():
            num_params += p.numel()
        print(Fmodel)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        #G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        #D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        #self.model.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

        model_path = os.path.join(self.model_save_dir,'200000-G.ckpt')
        print(model_path)
        self.Fmodel = torch.load(model_path, map_location=self.device)
        #self.G = torch.load(model_path, map_location=self.device)

        #self.Fmodel.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        #self.Fmodel.load_state_dict(torch.load(model_path, map_location=self.device))

        # model_path = os.path.join(self.model_save_dir, '200000.pth')
        # torch.save(self.model, model_path)


    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)


    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    _, _, _, xf = self.Fmodel(x_real, c_trg, c_org)
                    x_fake_list.append(xf)

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
