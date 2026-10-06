import numpy as np
import scipy.stats as ss

class MixtureOfGaussians(ss.rv_continuous):
    def _pdf(self, x):
        res = 0.4 * ss.norm(loc=0, scale=0.5).pdf(x) + 0.6 * ss.norm(loc=3, scale=0.5).pdf(x)
        return res
    
    def _cdf(self, x):
        res = 0.4 * ss.norm(loc=0, scale=0.5).cdf(x) + 0.6 * ss.norm(loc=3, scale=0.5).cdf(x)
        return res

def cal_ks_wd(generated, u, data_name):

    if data_name == "normal":
        true = ss.norm(2, 0.5)

    elif data_name == "exp":
        true = ss.expon(scale=2)
    
    elif data_name == "mix":
        true = MixtureOfGaussians(a=-np.inf, b=np.inf)

    elif data_name == "copula":
        true = ss.gamma(a = 2, loc=-2, scale=1)
        
    ks = ss.kstest(generated, true.cdf)[0]
    wd = ss.wasserstein_distance(u_values = generated, v_values = u)

    ks = round(ks, 3)
    wd = round(wd, 3)
    return ks, wd

def cal_lambda(generated):
    # Lambda function estimation
    points = np.linspace(0, 1, 400)

    # phi(t) / (dphi(t)/dt)
    true = -0.5 * (points - points**3)

    # phi(t) / (dphi(t)/dt) can be estimated by Kendall’s measure, that is, 
    # phi(t) / (dphi(t)/dt) = t - K_{C}(t) (An Introduction to Copulas, Nelsen, Theorem 4.3.4)
    u1 = generated[:, 0]
    u2 = generated[:, 1]

    n = len(u1)
    z = np.zeros(n)

    for i in range(n):
        z[i] = np.mean((u1 <= u1[i]) & (u2 <= u2[i]))
    
    res = ss.ecdf(z)
    estimated = points - res.cdf.evaluate(points) 

    # distances between true and estimated
    L1 = round((np.abs(true - estimated)/len(points)).sum(), 3)
    L2 = round((np.square(true - estimated)/len(points)).sum(), 3)
    
    return estimated, L1, L2