import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.cell2d import Cell, ConvBR

# Parameter-optimizable classifier used by neural archatecture search
# Classifier built from networks.cell2d Cell
# Cascade cells downsizing image size and increasing features finally connecting to fully connected layers
# NAS Cell parameters: out_channels - output chanels, steps - number of convolution interations
# NAS model parameters: nn.Linear output chanels
# Oblation study parameters: cell.residual, ConvBR.batch_norm, ConvBR.padding_mode
class AutoClassifier(nn.Module):
    def __init__(self, num_cells, cell=Cell):
        super(AutoClassifier, self).__init__()
        print('{}.{}'.format(__class__.__name__, __name__))

        self.num_cells = num_cells
        self.cell_dfn = cell
        self.cells = nn.ModuleList()
        self.parameters = []

        self.parameters.append({'name':'channels','tensor':(1e-3 * torch.randn(self.num_cells)).clone().detach().requires_grad_(True)})
        self.parameters.append({'name':'steps','tensor':(1e-3 * torch.randn(self.num_cells)).clone().detach().requires_grad_(True)})

        self.initialize()

    def forward(self, x):
        print('{}.{}'.format(__class__.__name__, __name__))
        return x

    def initialize(self):
        print('{}.{}'.format(__class__.__name__, __name__))

        # Registers neural architecture searchable parameters to PyTorch
        for i in range(self.parameters):
            self.register_parameter(self.parameters[i]['name'], torch.nn.Parameter(self.parameters[i]['tensor']))

