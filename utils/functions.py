import torch
import math
import numpy as np

def GaussianBasis(i, zero=0.0, sigma=0.33):
    return torch.exp(-1*torch.square((i-zero)/(2*sigma))) # torch.square not supported by torch.onnx

def NormGausBasis(len, i, depth, r=1.0):
        den = 0.0
        num = 0.0
        for j in range(len):
            bias = GaussianBasis(j,depth,r)
            if j==i:
                num=bias
            den = den + bias
        return num/den

def SigmoidScale(step, sigmoid_scale = 5, exp_scale = 0.1, maxscale=500):
    kSigmoid = sigmoid_scale+math.exp(exp_scale*step)
    return min(kSigmoid, maxscale)

# Exponential function from vertex and point
class Exponential():
    def __init__(self,vx=0.0, vy=0.0, px=1.0, py=1.0, power=2.0):
        self.vx = vx
        self.vy = vy
        self.px = px
        self.py = py
        if power < 0:
            raise ValueError('Exponential error power {} must be >= 0'.format(power))
        self.power = power
        if px <= vx:
            raise ValueError('Exponential error px={} must be > vx'.format(px, vx))
        else:
            self.a = (py-vy)/np.power(px-vx,power)
    def f(self, x):
        dx = x-self.vx
        y = self.a*np.power(x-self.vx,self.power) + self.vy
        return y