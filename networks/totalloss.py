import os, sys
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional
from enum import Enum

class FenceSitterEjectors(Enum):
    none = 'None'
    prune_basis = 'prune_basis'
    dais = 'dais'

    def __str__(self):
        return self.value

class TotalLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, isCuda, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 prune=None, reduction: str = 'mean', k_structure=0.0, target_structure=torch.as_tensor([1.0], dtype=torch.float32), 
                 class_weight=None, search_structure=True, k_prune_basis=1.0, k_prune_exp=3.0, sigmoid_scale=5.0, exp_scale=10, ejector=FenceSitterEjectors.none) -> None:
        super(TotalLoss, self).__init__(weight, size_average, prune, reduction)
        self.isCuda = isCuda
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.k_structure = k_structure
        self.softsign = nn.Softsign()
        self.target_structure = target_structure  
        self.archloss = nn.L1Loss()
        #self.archloss = nn.MSELoss()
        self.class_weight = class_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        self.search_structure = search_structure
        self.k_prune_basis = k_prune_basis
        self.k_prune_exp = k_prune_exp
        self.sigmoid_scale = sigmoid_scale
        self.exp_scale = exp_scale
        self.ejector = ejector

        if self.isCuda:
            if weight:
                self.weight = weight.cuda()
            self.target_structure = self.target_structure.cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, network) -> torch.Tensor:
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        cross_entropy_loss = 1.0*self.cross_entropy_loss(input, target.long())

        dims = []
        depths = []

        architecture_weights, total_trainable_weights, cell_weights = network.ArchitectureWeights()
        sigmoid_scale = torch.zeros(1, device=architecture_weights.device)
        prune_loss = torch.zeros(1, device=architecture_weights.device)
        if self.search_structure:
            architecture_reduction = architecture_weights/total_trainable_weights
            architecture_loss = self.k_structure*self.archloss(architecture_reduction,self.target_structure)

            if self.ejector == FenceSitterEjectors.prune_basis or self.ejector == FenceSitterEjectors.prune_basis.value:
                prune_basises = []
                for cell_weight in cell_weights:
                    for conv_weights in cell_weight['cell_weight']:
                        # conv_weights is from 0..1
                        # prune_weight is from 0..1
                        # weight is pruned if either cell weight < threshold or prunewight is < threshold.  
                        # Average is not a great model for this but is continuous where min is discontinuous
                        # Average will return a fewer than will be pruned
                        # Product will return more to prune that will be pruned
                        # Minimum is discontinuous and will shift the optimizer focuse from convolution to cell
                        #prune_basis = (conv_weights+cell_weight['prune_weight'])/2.0
                        prune_basis = conv_weights*cell_weight['prune_weight']
                        #prune_basis = conv_weights.minimum(cell_weight['prune_weight'])
                        prune_basises.extend(prune_basis)
                len_prune_basis = len(prune_basises)
                #architecture_exp = torch.exp(-1*self.k_prune_exp*architecture_loss/self.k_structure)
                if len_prune_basis > 0:
                    prune_basises = torch.stack(prune_basises)
                    prune_basis = torch.linalg.norm(prune_basises)/np.sqrt(len_prune_basis)
                    prune_loss = self.k_prune_basis*prune_basis           

            total_loss = cross_entropy_loss + architecture_loss + prune_loss
        else:
            architecture_loss = torch.zeros(1)
            architecture_reduction = torch.zeros(1)
            prune_loss = torch.zeros(1)
            total_loss = cross_entropy_loss
        return total_loss,  cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss, sigmoid_scale