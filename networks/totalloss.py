import torch
import torch.nn as nn
from typing import Callable, Optional


class TotalLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, isCuda, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 prune=None, reduction: str = 'mean', k_structure=0.0, target_structure=torch.as_tensor([1.0], dtype=torch.float32), 
                 class_weight=None, search_structure=True, k_prune_basis=1.0, k_prune_exp=3.0) -> None:
        super(TotalLoss, self).__init__(weight, size_average, prune, reduction)
        self.isCuda = isCuda
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.k_structure = k_structure
        self.softsign = nn.Softsign()
        self.target_structure = target_structure  
        #self.archloss = nn.L1Loss()
        self.archloss = nn.MSELoss()
        self.class_weight = class_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        self.search_structure = search_structure
        self.k_prune_basis = k_prune_basis
        self.k_prune_exp = k_prune_exp

        if self.isCuda:
            if weight:
                self.weight = weight.cuda()
            self.target_structure = self.target_structure.cuda()



    def forward(self, input: torch.Tensor, target: torch.Tensor, network) -> torch.Tensor:
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        cross_entropy_loss = 1.0*self.cross_entropy_loss(input, target.long())

        dims = []
        depths = []

        architecture_weights, total_trainable_weights, cell_weights, prune_basis = network.ArchitectureWeights()
        if self.search_structure:
            architecture_reduction = architecture_weights/total_trainable_weights
            architecture_loss = self.k_structure*self.archloss(architecture_reduction,self.target_structure)
            prune_loss = self.k_prune_basis*prune_basis*torch.exp(-1*self.k_prune_exp*architecture_loss)

            total_loss = cross_entropy_loss + architecture_loss + prune_loss
        else:
            architecture_loss = torch.zeros(1)
            architecture_reduction = torch.zeros(1)
            prune_loss = torch.zeros(1)
            total_loss = cross_entropy_loss
        return total_loss,  cross_entropy_loss, architecture_loss, architecture_reduction, cell_weights, prune_loss