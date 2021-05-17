import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from collections import namedtuple

Genotype = namedtuple('Genotype_2D', 'cell cell_concat')

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

class ConvBR(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, bn=True, relu=True):
        super(ConvBR, self).__init__()
        self.relu = relu
        self.use_bn = bn

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

PRIMITIVES = [
    'skip_connect',
    'conv_3x3']

OPS = {
    'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    'conv_3x3': lambda C, stride: ConvBR(C, C, 3, stride, 1)
}

class MixedOp(nn.Module):
    def __init(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier, prev_fmultiplier_same,filter_multiplier):
        super(Cell, self).__init__()

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self._prev_fmultiplier_same = prev_fmultiplier_same

        if prev_fmultiplier_same is not None:
            self.C_prev_same = int(prev_fmultiplier_same * block_multiplier)
            self.preprocess_same = ConvBR(self.C_prev_same, self.C_out, 1, 1, 0)

        if prev_prev_fmultiplier != -1:
            self.pre_preprocess = ConvBR(self.C_prev_prev, self.C_out, 1, 1, 0)

        self._steps = steps
        self.block_multiplier = block_multiplier
        self.pre_preprocess = ConvBR(self.C_prev_prev, self.C_out, 1, 1, 0)
        self._ops = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                if prev_prev_fmultiplier == -1 and j == 0:
                    op = None
                else:
                    op = MixedOp(self.C_out, stride)
                self._ops.append(op)

        self._initialize_weights()


    def forward(self, s0, s1, n_alphas):
        all_states = []

        if s0 is None:
            s0 = 0
        else:
        if s1 is None:
            s1 = 0
        else:

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0


        states = [s0, s1]
        final_concates = []

        offset = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if self._ops[branch_index] is None:
                    continue
                new_state = self._ops[branch_index](h, n_alphas[branch_index])                
                new_states.append(new_state)

            s = sum(new_states)
            offset += len(states)
            states.append(s)

            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates.append(concat_feature)
        return final_concates

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)