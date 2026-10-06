from loglikelihood import *

from functools import partial
import math
import torch
from torch import nn

class Custom_Layer(nn.Module):
    def __init__(self, in_features, out_features, 
                 LayerNorm, SkipConnection) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.norm_layer = nn.LayerNorm(out_features) if LayerNorm else nn.Identity()
        self.act = nn.PReLU() 
        
        self.use_skip = bool(SkipConnection)
        if self.use_skip:
            self.linear_2 = nn.Linear(out_features, out_features, bias = False)
            self.skip_connection_layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        # reference DOI: 10.1109/MSP.2023.3286988
        res = self.linear_1(x)
        res = self.norm_layer(res)
        res = self.act(res)

        if self.use_skip:
            res = self.linear_2(res) + self.skip_connection_layer(x)

        return res

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.h_dims = [self.x_dim] + list(reversed(self.h_dims))

        layers = []
        for i in range(len(self.h_dims) - 1):
            layers.append(Custom_Layer(self.h_dims[i], self.h_dims[i+1], self.LayerNorm, self.SkipConnection_encoder))
        
        if self.SkipConnection_encoder:
            self.mu = Custom_Layer(self.h_dims[-1], self.z_dim, self.LayerNorm, self.SkipConnection_encoder)
            self.logvar = Custom_Layer(self.h_dims[-1], self.z_dim, self.LayerNorm, self.SkipConnection_encoder)
        else:
            self.mu = nn.Linear(self.h_dims[-1], self.z_dim)
            self.logvar = nn.Linear(self.h_dims[-1], self.z_dim)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x : [batch_size, 1, x_dim] - > h : [batch_size, 1, h_dims_last]
        h = self.encoder(x)

        # h : [batch_size, 1, h_dims_last] - > mu, logvar : [batch_size, 1, z_dim]     
        mu = self.mu(h)
        logvar = self.logvar(h) # torch.clamp if needed
        return mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)   
        
        self.h_dims = [self.z_dim] + self.h_dims

        layers = []
        for i in range(len(self.h_dims) - 1):
            layers.append(Custom_Layer(self.h_dims[i], self.h_dims[i+1], self.LayerNorm, self.SkipConnection_decoder))
        
        if self.SkipConnection_decoder:
            layers.append(Custom_Layer(self.h_dims[-1], self.u_dim, self.LayerNorm, self.SkipConnection_decoder))
        else:
            layers.append(nn.Linear(self.h_dims[-1], self.u_dim))
        
        if self.data_name == 'cortex':
            layers.append(nn.Softmax(dim=-1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        # z : [batch_size, num_samples , z_dim] -> u : [batch_size, num_samples, u_dim]
        u = self.decoder(z)
        return u

class Model(nn.Module):
    def __init__(self, std, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value) 
        
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)
        
        if self.data_name in ("normal", "exp", "mix"):
            self.sigma2 = 1 / (std)**2
            self.sigma2 = torch.from_numpy(self.sigma2).to(self.device)
            self.cal_ll = partial(ll_normal, sigma2 = self.sigma2)
        
        elif self.data_name in ("copula"):
            self.fix_dim = 5
            self.rand_dim = 5
            self.linear = nn.Linear(self.fix_dim, 1, bias = False)
            self.cal_ll = partial(ll_poisson_reg, linear = self.linear, fix_dim = self.fix_dim, rand_dim = self.rand_dim)
        
        elif self.data_name in ("cortex"):
            self.log_inv_disp = nn.Parameter(torch.zeros(1, 1, self.u_dim))
            self.cal_ll = partial(ll_negativebinomial, log_inv_disp = self.log_inv_disp)
        
        elif self.data_name in ("VerbAgg"):
            self.cal_ll = ll_bernoulli_reg

    def VAE_forward(self, x):
        # x : [batch_size, 1, x_dim] -> mu, logvar : [batch_size, 1, z_dim]
        mu, logvar = self.encoder(x)

        # std : [batch_size, 1, z_dim] -> std, eps : [batch_size, num_samples, z_dim]
        std = torch.exp(0.5 * logvar).expand(-1, self.num_samples, -1)
        eps = torch.randn_like(std)

        # z : [batch_size, num_samples, z_dim]
        z = mu + std * eps

        # z : [batch_size, num_samples, z_dim] -> u : [batch_size, num_samples, u_dim]
        u = self.decoder(z)
        
        # kl : [batch_size, 1] with torch.mean
        kl = torch.mean(-0.5 * torch.sum(1.0 + logvar - torch.square(mu) - torch.exp(logvar), dim=2))

        # ll : [batch_size, num_samples] with torch.mean
        ll = torch.mean(self.cal_ll(x, u))

        # loss
        elbo = ll - kl
        return ll, kl, -elbo

    def IWAE_forward(self, x):
        # x : [batch_size, 1, x_dim] -> mu, logvar : [batch_size, 1, z_dim]
        mu, logvar = self.encoder(x)

        # std : [batch_size, 1, z_dim] -> std, eps : [batch_size, num_samples, z_dim]
        std = torch.exp(0.5 * logvar).expand(-1, self.num_samples, -1)
        eps = torch.randn_like(std)

        # z : [batch_size, num_samples, z_dim]
        z = mu + std * eps

        # z : [batch_size, num_samples, z_dim] -> u : [batch_size, num_samples, u_dim]
        u = self.decoder(z)
        if self.data_name == "cortex":
            u = u * torch.sum(x, dim=2, keepdim=True)

        # log q(z|x) : [batch_size, num_samples]
        log_qzx = -0.5 * torch.sum(torch.square(z - mu) * torch.exp(-logvar) + logvar + math.log(2 * math.pi), dim=2)

        # log p(z) : [batch_size, num_samples]
        log_pz = -0.5 * torch.sum(torch.square(z) + math.log(2 * math.pi), dim=2)

        # log p(x|z) : [batch_size, num_samples]
        log_pxz = self.cal_ll(x, u)

        # log w : [batch_size, num_samples]
        log_w = log_pxz + self.kl_weight * (log_pz - log_qzx)

        # loss : [batch_size] with torch.mean
        elbo = torch.mean(torch.logsumexp(log_w, dim=1) - math.log(self.num_samples))
        return 0, 0, -elbo

    def IWRCL_forward(self, x): 
        # x : [batch_size, 1, x_dim] -> mu, logvar : [batch_size, 1, z_dim]
        mu, logvar = self.encoder(x)

        # std : [batch_size, 1, z_dim] -> std, eps : [batch_size, num_samples, z_dim]
        std = torch.exp(0.5 * logvar).expand(-1, self.num_samples, -1)
        eps = torch.randn_like(std)

        # z : [batch_size, num_samples, z_dim]
        z = mu + std * eps

        # z : [batch_size, num_samples, z_dim] -> u : [batch_size, num_samples, u_dim]
        u = self.decoder(z)
        if self.data_name == "cortex":
            u = u * torch.sum(x, dim=2, keepdim=True)

        # log q(z|x) : [batch_size, num_samples]
        log_qzx = -0.5 * torch.sum(torch.square(z - mu) * torch.exp(-logvar) + logvar + math.log(2 * math.pi), dim=2)

        # log p(z) : [batch_size, num_samples]
        log_pz = -0.5 * torch.sum(torch.square(z) + math.log(2 * math.pi), dim=2)

        # log p(x|z) : [batch_size, num_samples]
        log_pxz = self.cal_ll(x, u)

        # log w : [batch_size, num_samples]
        log_w = (log_pxz + log_pz - log_qzx).detach()
        w = torch.softmax(log_w, dim=1)

        # ll : [batch_size,] with torch.mean
        ll = torch.mean(torch.sum(w * log_pxz, dim=1))

        # kl : [batch_size, 1] with torch.mean
        kl = torch.mean(-0.5 * torch.sum(1.0 + logvar - torch.square(mu) - torch.exp(logvar), dim=2))

        # loss
        elbo = ll - self.kl_weight * kl
        return ll, kl, -elbo