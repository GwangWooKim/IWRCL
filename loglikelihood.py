import math
import torch 
import torch.nn.functional as F

# x = [batch_size, 1, x_dim]
# u = [batch_size, num_samples, u_dim]
# ll = [batch_size, num_samples]

def ll_normal(x, u, sigma2):
    ll = -0.5 * (torch.square(x - u) / sigma2 + torch.log(sigma2) + math.log(2 * math.pi))
    return torch.sum(ll, dim=2)

def ll_poisson_reg(data, u, linear, fix_dim = 5, rand_dim = 5):
    x_end = fix_dim
    w_end = fix_dim + rand_dim
    y_end = fix_dim + rand_dim + 1

    x, w, y = data[:, :, :x_end], data[:, :, x_end:w_end], data[:, :, w_end:y_end]
    log_lambda = linear(x) + torch.sum(w * u, dim=2, keepdim=True)

    ll = y * log_lambda - torch.exp(log_lambda) - torch.lgamma(y+1)
    return ll.squeeze(2)

def ll_negativebinomial(x, u, log_inv_disp):
    inv_disp = torch.exp(log_inv_disp)
    ll = (
    x * torch.log(u / inv_disp) - x * torch.log(1 + u / inv_disp) - inv_disp * torch.log(1 + u / inv_disp)
    + torch.lgamma(x + inv_disp) - torch.lgamma(inv_disp) - torch.lgamma(x + 1)
    )
    return torch.sum(ll, dim=2)

def ll_bernoulli_reg(data, u, fix_dim = 0, rand_dim = 5):
    x_end = fix_dim
    w_end = fix_dim + rand_dim
    y_end = fix_dim + rand_dim + 1

    w, y = data[:, :, x_end:w_end], data[:, :, w_end:y_end]
    y_logit = torch.sum(w * u, dim=2, keepdim=True)
    ll = y_logit * (y - 1) + F.logsigmoid(y_logit)
    return ll.squeeze(2)
