import torch
import torch.nn as nn
from typing import Callable, Optional


class TotalLoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, isCuda, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 prune=None, reduction: str = 'mean', k_structure=0.0, target_structure=torch.as_tensor([1.0], dtype=torch.float32), class_weight=None, search_structure=True) -> None:
        super(TotalLoss, self).__init__(weight, size_average, prune, reduction)
        self.isCuda = isCuda
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.k_structure = k_structure
        self.softsign = nn.Softsign()
        self.target_structure = target_structure  
        self.mseloss = nn.MSELoss()
        self.class_weight = class_weight
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.class_weight, ignore_index=self.ignore_index, reduction=self.reduction)
        self.search_structure = search_structure

        if self.isCuda:
            if weight:
                self.weight = weight.cuda()
            self.target_structure = self.target_structure.cuda()



    def forward(self, input: torch.Tensor, target: torch.Tensor, network) -> torch.Tensor:
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        loss = self.cross_entropy_loss(input, target.long())

        dims = []
        depths = []
        architecture_weights, total_trainable_weights = network.ArchitectureWeights()
        arcitecture_reduction = architecture_weights/total_trainable_weights
        architecture_loss = self.mseloss(arcitecture_reduction,self.target_structure)

        total_loss = loss
        if self.search_structure:
            total_loss += self.k_structure*architecture_loss
        return total_loss,  loss, arcitecture_reduction