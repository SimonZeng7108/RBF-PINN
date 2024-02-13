# RBF kernel functions
import torch
def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def gaussian_compact(alpha, h=3):
    mask = torch.ones_like(alpha)
    mask[alpha>h] = 0
    phi = torch.exp(-1*alpha.pow(2))
    phi = phi*mask
    return phi

def linear(alpha):
    phi = alpha
    return phi

def cubic(alpha):
    phi = alpha.pow(3)
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi