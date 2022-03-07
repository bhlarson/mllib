import torch

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