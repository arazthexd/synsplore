import torch
import torch.nn as nn
import torch.nn.functional as F

class SynsploreClassifier(nn.Module):
    def __init__(self, d_model: int, n_actions: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_actions)
        )
    
    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc (torch.Tensor): (N_Batch, max(L_Seq)-1, D_Model)

        Returns:
            torch.Tensor: (N_Batch, max(L_Seq)-1, N_Actions)
        """
        return self.mlp.forward(enc)
    
    def get_loss(self, enc: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enc (torch.Tensor): (N_Batch, max(L_Seq)-1, D_Model)
            true (torch.Tensor): (N_Batch, max(L_Seq)-1, N_Actions)

        Returns:
            torch.Tensor: (, )
        """
        ignore_mask = true.sum(dim=-1) == 0
        
        logits = self.forward(enc)
        logits = logits[~ignore_mask]
        return F.cross_entropy(logits, true[~ignore_mask]) 

if __name__ == "__main__":

    import random

    true = torch.zeros((10, 8, 4))
    for i in range(10):
        n = random.randint(0, 5)
        true[i, torch.arange(n, 8), torch.randint(0, 4, (8-n, ))] = 1
    x = SynsploreClassifier(128, 4).get_loss(torch.randn((10, 8, 128)), true)
    print(x, true.sum())

        
