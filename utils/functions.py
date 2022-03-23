import torch
import math

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

#def SigmoidScale(err, sigmoid_scale = 5, k_prune_exp=5, exp_scale = 10):
#    kSigmoid = sigmoid_scale+math.exp(k_prune_exp*(1-exp_scale*err))
#    return kSigmoid

def SigmoidScale(step, sigmoid_scale = 5, exp_scale = 0.1, maxscale=500):
    kSigmoid = sigmoid_scale+math.exp(exp_scale*step)
    return min(kSigmoid, maxscale)