import torch
import torch.nn as nn

import model.utils as utils

import torchcde
import mdn

class CDEFunc(nn.Module):
    '''
    Neural CDE grad function.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, input_dim, latent_dim):
        super(CDEFunc, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(latent_dim, 1024)
        self.relu1 = nn.PReLU()
        self.linear2 = nn.Linear(1024, input_dim * latent_dim)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(input_dim * latent_dim, input_dim * latent_dim)
    
    def forward(self, t, z):
        z = self.linear1(z)
        z = self.relu1(z)
        z = self.linear2(z)
        z = self.tanh(z) # important!
        z = self.linear3(z)

        z = z.view(z.shape[0], self.latent_dim, self.input_dim)

        return z



class CDE_MDN(nn.Module):
    '''
    A Neural CDE Mixture Density Network.
    '''
    def __init__(self, input_dim, latent_dim, output_dim):
        super(CDE_MDN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_gaussian = 12

        self.cde_func = CDEFunc(input_dim, latent_dim)
        self.initial = nn.Sequential(
            utils.create_net(input_dim, latent_dim, n_layers=1, n_units=1024, nonlinear=nn.PReLU),
        )
        self.readout = nn.Sequential(
            utils.create_net(latent_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU) for i in range(3)],
            mdn.MDN(in_features=1024, out_features=self.output_dim, num_gaussians=self.n_gaussian)
        )
        
    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              t=X.interval,
                              adjoint=False,
                              method="dopri5", rtol=1e-5, atol=1e-7)

        z_T = z_T[:, -1]
        pi, sigma, mu = self.readout(z_T).reshape(-1, self.output_dim, 3, self.n_gaussian)

        return pi, sigma, mu
    
    def mdn_loss(pi, sigma, mu, labels):
        return mdn.mdn_loss(pi, sigma, mu, labels)
