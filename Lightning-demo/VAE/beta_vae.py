import torch 
from base_model import BaseVAE
from torch.nn import functional as F 
from torch import nn 


class BetaVAE(BaseVAE):

    num_iter = 0 

    def __init__(self, in_channels, latent_dim,
                 hidden_dims=None, beta=4, gamma=1000, max_capacity=25,
                 Capacity_max_iter=1e5, loss_type='B', **kwargs):
        
        super(BetaVAE, self).__init__()
        self.beta = beta 
        self.latent_dim = latent_dim
        self.beta = beta 
        self.gamma = gamma 
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = [] 
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim 
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims)-1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2,padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU()
            ))
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
                    nn.Tanh()
                    )
    
    def encoder(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu 
    
    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
    def loss_function(self, *args, **kwargs):
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        recon_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0,5 * torch.sum(1+log_var - mu **2 - log_var.exp(), dim=1), dim=0)
        if self.loss_type == 'H':
            loss = recon_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(input.device)
            C  = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        return {'loss':loss, 'Reconstruction_Loss':recon_loss, 'KLD':kld_loss}
    def sample(self, num_samples, current_device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples 
    
    def generate(self, x, **kwargs):
        return self.forward(x)[0]
