import numpy as np

import torch
from torch import nn
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FullyConnectedAutoencoder(nn.Module):
  

    def __init__(self, layer_widths, activation_function, seed=1):
        super(FullyConnectedAutoencoder, self).__init__()
        
        # set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        self.layer_widths = layer_widths
        self.activation_function = activation_function
        
        w = self.layer_widths
        self.encode_layers = nn.ModuleList([nn.Linear(w[i], w[i+1]) for i in range(0, len(w)-1, 1)])
        self.decode_layers = nn.ModuleList([nn.Linear(w[i], w[i-1]) for i in range(len(w)-1, 0, -1)])


    def encode(self, x):
        
        # State s is input data x
        s = x
        
        # notational convenience
        widths = self.encode_layers
        a = self.activation_function
        
        # Iterate encoder layers
        for w in widths:
            s = a(w(s))
            
        # Code c
        c = s
        
        return c

    def decode(self, c):
        
        # State s is code data h
        s = c
        
        # notational convenience
        widths = self.decode_layers
        a = self.activation_function

        # Iterate decoder layers
        for w in widths[:-1]:
            s = a(w(s))

        # Reconstruction x_r
        x_r = widths[-1](s)
        
        return x_r
    
    
    def forward(self, x):
        
        return self.decode(self.encode(x))

    
    def train_model(self, lr, batch_size, num_epochs, train_dataset, test_dataset):

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = len(list(test_dataset)), shuffle = False, drop_last = True)
        
        self.train()
        
        print("Epoch MSELoss(train) MSELoss(test)")
        for epoch in range(num_epochs):

            epoch_loss_train = 0
            for i, batch_sample in enumerate(train_loader):
        
                x_batch, y_batch = batch_sample

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
        
                prediction = self(x_batch)

                l = self.loss(prediction, y_batch)
                epoch_loss_train += l.item()

                self.zero_grad()
                l.backward()
                self.optimizer.step()
            epoch_loss_train /= len(train_loader)
            
            if epoch % 100 == 0:
                epoch_loss_test = 0
                for i, batch_sample in enumerate(test_loader):
        
                    x_batch, y_batch = batch_sample

                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
        
                    prediction = self(x_batch)
        
                    l = self.loss(prediction, y_batch)
                    epoch_loss_test += l.item()
                epoch_loss_test /= len(test_loader)
            
                print(str(epoch) + " " + "{:.6f}".format(epoch_loss_train) + " " + "{:.6f}".format(epoch_loss_test))


    def save_model(self, path):

        torch.save(self.state_dict(), path + ".pt")


    def load_model(self, path):        
        
        self.load_state_dict(torch.load(path + ".pt"))
        self.eval()
        return self