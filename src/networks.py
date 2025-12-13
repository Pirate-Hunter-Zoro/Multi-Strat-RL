import torch
import torch.nn as nn
from src.config import AblationConfig, HIDDEN_FEATURES
import math

class RainbowDQN(nn.Module):
    
    def __init__(self, state_dim: int, num_actions: int, num_atoms: int, config: AblationConfig, use_cnn: bool=False):
        """
        Initialization for a RainbowDQN agent
        
        :param state_dim: Dimensions of the state
        :type state_dim: int
        :param num_actions: Number of possible actions
        :type num_actions: int
        :param num_atoms: Number of atoms to use in the case of C51
        :type num_atoms: int
        :param config: Describes which DQN techniques to use
        :type config: AblationConfig
        :param use_cnn: Flag for whether or not a cnn should be used
        :type use_cnn: bool
        """
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.use_cnn = use_cnn
        
        # The following will be used regardless of whether a cnn is used
        self.fc_rest = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_FEATURES, out_features=HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_FEATURES, out_features=num_actions * (1 if not config.use_distributional else num_atoms))
        )
        
        # Create convolutional layers if that is to be used
        if self.use_cnn:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),
                nn.ReLU()
            )
            self.initialized = False
            self.fc_start = None
        else:    
            self.fc_start = nn.Linear(in_features=state_dim, out_features=HIDDEN_FEATURES)
        
    def forward(self, X: torch.tensor) -> torch.tensor:
        """NOTE - one must softmax on the last dimension when returning the output

        Args:
            X (torch.tensor): Input observations

        Returns:
            torch.tensor: Processed output
        """
        if self.use_cnn:
            pixels = X.shape[1] // 3
            width = int(math.sqrt(pixels))
            X = X.view(-1, width, width, 3) # batch, width, width, channels
            X = X.permute(0, 3, 1, 2) # batch, channels, width, width
            X = self.conv(X)
            X = X.flatten(start_dim=1)
            if not self.initialized:
                n_features = X.shape[1]
                self.fc_start = nn.Linear(in_features=n_features, out_features=HIDDEN_FEATURES).to(X.device)
                self.initialized = True
        else:
            X = X.flatten(start_dim=1) # batching consistency
            
        batch = X.shape[0]
        X = self.fc_start(X)
        X = self.fc_rest(X)
        if self.config.use_distributional:
            X = X.reshape(shape=(batch, self.num_actions, self.num_atoms))
        return X