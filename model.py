from torch import nn, optim
from torch.nn import functional as F
import torch


class VAE(nn.Module):

    def __init__(self, len_):
        super(VAE, self).__init__()
        self.len_ = len_
        hidden_layer = 32  # 갯수 수정 필요 아마 128
        self.fc1 = nn.Linear(len_, hidden_layer)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_layer)

        self.fc21 = nn.Linear(hidden_layer, hidden_layer * 2)
        self.fc22 = nn.Linear(hidden_layer, hidden_layer * 2)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_layer * 2)

        self.fc_code = nn.Linear(hidden_layer * 2, hidden_layer * 2)

        self.fc3 = nn.Linear(hidden_layer * 2, hidden_layer)
        self.bn3 = nn.BatchNorm1d(num_features=hidden_layer)

        self.fc4 = nn.Linear(hidden_layer, len_)
        self.bn4 = nn.BatchNorm1d(num_features=len_)

        self.bn = nn.BatchNorm1d(num_features=hidden_layer * 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        h1 = self.fc1(x)
        # h1 = self.bn1(h1) # batchnormalization: h1
        h1 = F.relu(h1)

        h21 = self.fc21(h1)
        # h21 = self.bn2(h21) # batchnormalization: h21(optional)

        h22 = self.fc22(h1)
        # h22 = self.bn2(h22) # batchnormalization: h22(optional)
        return h21, h22

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # std 내에서 정수를 균등하게 생성
        return mu + eps * std

    # Latent_coded의 act.fn 수정 필요
    def latent_coded(self, z):
        h_coded = torch.sigmoid(self.fc_code(z))  # F.sigmoid(self.fc_code(z))
        return h_coded

    def decode(self, z):
        h3 = self.fc3(z)
        # h3 = self.bn3(h3)
        h3 = F.relu(h3)
        # torch.div(h3 - h3.mean(axis=1).view(-1,1).repeat(1, h3.shape[1]), h3.std(axis=1).view(-1,1).repeat(1, h3.shape[1])) # batch norm.
        h4 = self.fc4(h3)
        # h4 = self.bn4(h4)
        h4 = F.relu(h4)
        return h4

    def forward(self, x):
        # len_ = 64
        #       hidden_layer = 200
        mu, logvar = self.encode(x.view(-1, self.len_))
        z = self.reparameterize(mu, logvar)
        # print("reparameterize:", z.shape)
        # z = self.bn(z) # bn
        z = self.latent_coded(z)

        z_coded = (z >= 0.5).int().float()
        # print("z_coded:", z_coded.shape)
        return self.decode(z_coded), z_coded, mu, logvar, z
#        return self.decode(z), z, mu, logvar

def loss_function(recon_x, x, mu, logvar, z, z_coded):
    recon_x = recon_x.double()
    x = x.double()
    # Loss term1: BCE to RMSE
    loss_1 = torch.sqrt(torch.mean((recon_x - x) ** 2))
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, len_), reduction='mean')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    loss_2 = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # MSE = torch.mean(torch.mean((z - z_coded).pow(2),axis=1))
    loss_3 = torch.mean(torch.mean((z - z_coded).abs(), axis=1))
    return loss_1 + loss_2 + loss_3