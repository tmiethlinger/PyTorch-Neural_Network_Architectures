# Reference: Ardizzone L. "Analyzing Inverse Problems with Invertible Neural Networks"

import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import grad

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InvertibleNeuralNetwork(nn.Module):

    def __init__(self, dimensions, activation, loss_weights, seed=1):
        super(InvertibleNeuralNetwork, self).__init__()

        # dimensions of INN
        nx, ny, nz, n, width, depth, blocks = dimensions

        # dimensions of input, output and latent space
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n = n

        # dimensions of subnet
        self.width = width
        self.depth = depth
        self.blocks = blocks

        # relative weighting of losses:
        w_y, w_z, w_x1, w_x2 = loss_weights
        self.w_y = w_y
        self.w_z = w_z
        self.w_x1 = w_x1
        self.w_x2 = w_x2

        # activation function
        self.activation = activation

        # reproducibility
        torch.manual_seed(seed)

        # Set up subnetwork
        def subnet_fc(c_in, c_out):
            layers = []
            for i in range(self.depth-1):
                if i == 0:
                    layers.append(nn.Linear(c_in, self.width))
                else:
                    layers.append(nn.Linear(self.width, self.width))
                layers.append(self.activation)
            layers.append(nn.Linear(self.width, c_out))
            return nn.Sequential(*layers)

        nodes = [InputNode(self.n, name='input')]

        for k in range(self.blocks):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':subnet_fc, 'clamp':2.0},
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1],
                              PermuteRandom,
                              {'seed':k},
                              name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        self.model = ReversibleGraphNet(nodes, verbose=False)

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

        # losses
        self.loss_y = torch.nn.MSELoss()
        self.loss_z = self.MMD_multiscale
        self.loss_x1 = self.MMD_multiscale
        self.loss_x2 = torch.nn.MSELoss()

        # Initialize network parameter
        for param in self.trainable_parameters:
            param.data = 0.1*torch.randn_like(param)
        self.model.to(device);
        return


    def forward(self, x):

        x_pad = torch.cat((x, torch.zeros(x.shape[0], self.n-x.shape[1], device=device)), dim=1)
        y_pad_z = self.model(x_pad)[0]
        return y_pad_z


    def inverse(self, yz):

        if yz.shape[1] < self.n:
            y_pad_z = torch.cat((yz[:, :self.ny],
                                 torch.zeros(yz.shape[0], self.n-yz.shape[1], device=device),
                                 yz[:, -self.nz:]), dim=1)
        else:
            y_pad_z = yz
        x_pad = self.model(y_pad_z, rev=True)[0]
        return x_pad


    def MMD_multiscale(self, x, y):

        xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() - 2.*xx + rx
        dxy = rx.t() - 2.*xy + ry
        dyy = ry.t() - 2.*yy + ry

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        for a in [0.04, 0.16, 0.64]:
            XX += a**2 / (a**2 + dxx)
            XY += a**2 / (a**2 + dxy)
            YY += a**2 / (a**2 + dyy)

        return torch.mean(XX - 2*XY + YY)


    def train_model(self, lr, batch_size, n_epochs, train_dataset, test_dataset):

        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, eps=1e-6)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last = True)


        print("Epoch Loss(train) Loss(test) L_y L_z L_x1 L_x2")

        for epoch in range(n_epochs):

            # Train inn
            self.train()

            L = 0
            L_y = 0
            L_z = 0
            L_x1 = 0
            L_x2 = 0

            for x, y in train_loader:

                #Turn
                self.optimizer.zero_grad()

                # Forward step:
                pad_yz = torch.zeros(self.batch_size, self.n-self.ny-self.nz, device=device)
                y_pad_z = torch.cat((y, pad_yz, torch.randn(self.batch_size, self.nz, device=device)), dim=1)
                pad_x = torch.zeros(self.batch_size, self.n-self.nx, device=device)
                x_pad = torch.cat((x, pad_x), dim=1)

                y_pad_z_pred = self(x_pad)

                l_y = self.w_y * self.loss_y(y_pad_z[:, :-self.nz], y_pad_z_pred[:, :-self.nz])

                l_z = self.w_z * self.loss_z(y_pad_z[:, -self.nz:], y_pad_z_pred[:, -self.nz:])

                l_forward = l_y + l_z
                l_forward.backward(retain_graph=True)

                # Backward step:
                pad_yz = torch.zeros(self.batch_size, self.n-self.ny-self.nz, device=device)
                y_pad_z = torch.cat((y, pad_yz, y_pad_z_pred[:, -self.nz:]), dim=1)
                y_pad_z_rand = torch.cat((y, pad_yz, torch.randn(self.batch_size, self.nz, device=device)), dim=1)

                x_pad_pred_rand = self.inverse(y_pad_z_rand)
                l_x1 = self.w_x1 * self.loss_x1(x_pad_pred_rand[:, :self.nx], x_pad[:, :self.nx])

                x_pad_pred = self.inverse(y_pad_z)
                l_x2 = self.w_x2 * self.loss_x2(x_pad_pred, x_pad)

                l_inverse = l_x1 + l_x2
                l_inverse.backward()
                self.optimizer.step()

                L_y += l_y.data.item()
                L_z += l_z.data.item()
                L_x1 += l_x1.data.item()
                L_x2 += l_x2.data.item()

            L = L_y + L_z + L_x1 + L_x2
            L_train = L / len(train_loader)

            # Compute loss of test set
            if epoch % 10 == 0:
                self.model.eval()

                L = 0
                L_y = 0
                L_z = 0
                L_x1 = 0
                L_x2 = 0

                for x, y in test_loader:

                    # Forward error:
                    pad_yz = torch.zeros(self.batch_size, self.n-self.ny-self.nz, device=device)
                    y_pad_z = torch.cat((y, pad_yz, torch.randn(self.batch_size, self.nz, device=device)), dim=1)
                    pad_x = torch.zeros(self.batch_size, self.n-self.nx, device=device)
                    x_pad = torch.cat((x, pad_x), dim=1)

                    y_pad_z_pred = self(x_pad)

                    l_y = self.w_y * self.loss_y(y_pad_z[:, :-self.nz], y_pad_z_pred[:, :-self.nz])

                    l_z = self.w_z * self.loss_z(y_pad_z[:, -self.nz:], y_pad_z_pred[:, -self.nz:])

                    # Backward error:
                    pad_yz = torch.zeros(self.batch_size, self.n-self.ny-self.nz, device=device)
                    y_pad_z = torch.cat((y, pad_yz, y_pad_z_pred[:, -self.nz:]), dim=1)
                    y_pad_z_rand = torch.cat((y, pad_yz, torch.randn(self.batch_size, self.nz, device=device)), dim=1)

                    x_pad_pred_rand = self.inverse(y_pad_z_rand)
                    l_x1 = self.w_x1 * self.loss_x1(x_pad_pred_rand[:, :self.nx], x_pad[:, :self.nx])

                    x_pad_pred = self.inverse(y_pad_z)
                    l_x2 = self.w_x2 * self.loss_x2(x_pad_pred, x_pad)

                    L_y += l_y.data.item()
                    L_z += l_z.data.item()
                    L_x1 += l_x1.data.item()
                    L_x2 += l_x2.data.item()

                L = L_y + L_z + L_x1 + L_x2
                L_test = L / len(test_loader)
                L_y = L_y / len(test_loader)
                L_z = L_z / len(test_loader)
                L_x1 = L_x1 / len(test_loader)
                L_x2 = L_x2 / len(test_loader)

                print(str(epoch) + " " +
                      "{:.5f}".format(L_train) + " " + "{:.5f}".format(L_test) + " " +
                      "{:.5f}".format(L_y) + " " + "{:.5f}".format(L_z) + " " +
                      "{:.5f}".format(L_x1) + " " + "{:.5f}".format(L_x2))


    def save_model(self, path):

        torch.save(self.model.state_dict(), path + ".pt")


    def load_model(self, path):

        self.model.load_state_dict(torch.load(path + ".pt"))
        self.model.eval()
        return self(base)
