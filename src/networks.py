import torch
import torch.nn as nn
from src.config import AblationConfig, HIDDEN_FEATURES

class RainbowDQN(nn.Module):
    
    def __init__(self, state_dim: int, num_actions: int, num_atoms: int, config: AblationConfig):
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.mlp = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_FEATURES, out_features=num_actions * (1 if not config.use_distributional else num_atoms))
        )
        
    def forward(self, X: torch.tensor) -> torch.tensor:
        """NOTE - one must softmax on the last dimension when returning the output

        Args:
            X (torch.tensor): Input observations

        Returns:
            torch.tensor: Processed output
        """
        X = X.flatten(start_dim=1)
        batch = X.shape[0]
        X = self.mlp(X)
        if self.config.use_distributional:
            X = X.reshape(shape=(batch, self.num_actions, self.num_atoms))
        return X