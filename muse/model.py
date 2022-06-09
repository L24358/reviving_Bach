import torch
import torch.nn as nn

class cnn_varencoder(nn.Module):
    def __init__(self, inp_channel, mult):
        super(cnn_varencoder, self).__init__()
        
        self.conv1 = nn.Conv1d(inp_channel, 16, 21, 1, 0)
        self.conv21 = nn.Conv1d(16, 4, 11, 1, 0)
        self.conv22 = nn.Conv1d(16, 4, 11, 1, 0)
        self.pool = nn.MaxPool1d(2,2)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(160+200*(mult-1), 20)
        
    def forward(self, I):
        h = self.pool(self.relu(self.conv1(I)))
        h1, h2 = self.pool(self.relu(self.conv21(h))), self.pool(self.relu(self.conv22(h)))
        h1, h2 = self.linear(self.flat(h1)), self.linear(self.flat(h2))
        return h1, h2
    
class cnn_vardecoder(nn.Module):
    def __init__(self, inp_channel, mult, p=0):
        super(cnn_vardecoder, self).__init__()        
        self.tconv1 = nn.ConvTranspose1d(4, 16, 11, 1, 0)
        self.tconv2 = nn.ConvTranspose1d(16, inp_channel, 31, 1, 0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(20, (160+200*(mult-1))*4)
        self.dropout = nn.Dropout(p=p)
        
    def forward(self, z):
        h = self.relu(self.linear(z))
        h = h.view(1,4,-1)
        h = self.relu(self.tconv1(h))
        h = self.tconv2(h)
        h = self.dropout(h)
        return h
    
class cnn_varautoencoder(nn.Module):
    def __init__(self, inp_channel, mult, p=0):
        super(cnn_varautoencoder, self).__init__()
        self.encoder = cnn_varencoder(inp_channel, mult)
        self.decoder = cnn_vardecoder(inp_channel, mult, p=p)
        self.use_var = False
        self.mu = None
        self.var = None
        self.mult = mult
        
    def reparameterize(self, mu, logvar):
        '''mu: mean matrix. logvar: variance matrix.'''
        if self.training or self.use_var:
            std = torch.exp(logvar/2)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:       
            return mu
        
    def forward(self, I):
        mu, logvar = self.encoder(I)
        z = self.reparameterize(mu, logvar)
        z = z.view(1,1,-1)
        y = self.decoder(z)
        return y, mu, logvar