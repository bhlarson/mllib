import torch.nn as nn
import torch.nn.functional as F
from networks.cell_2d import Cell, ConvBR

# Parameter-optimizable classifier used by neural archatecture search
# Classifier built from networks.cell_2d Cell
# Cascade cells downsizing image size and increasing features finally connecting to fully connected layers
# NAS Cell parameters: out_channels - output chanels, steps - number of convolution interations
# NAS model parameters: nn.Linear output chanels
# Oblation study parameters: cell.residual, ConvBR.batch_norm, ConvBR.padding_mode
class AutoClassifier(nn.Module):
    def __init__(num_layers, cell=Cell):
        super(AutoClassifier, self).__init__()
        print('{}.{}'.format(__class__.__name__, __name__))

        self.cells = nn.ModuleList()
        self._num_layers = num_layers

    def forward(self, x):
        print('{}.{}'.format(__class__.__name__, __name__))
        return x

    def initialize(self):
        print('{}.{}'.format(__class__.__name__, __name__))