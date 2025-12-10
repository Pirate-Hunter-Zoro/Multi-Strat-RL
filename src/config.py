import torch
from enum import Enum

NUM_FRAMES = 200000
HIDDEN_FEATURES = 512
PRINT_EVERY = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
DEBUG = False

class AblationConfig:
        
        def __init__(self, use_distributional: bool=False, use_delayed: bool=False, use_magnet: bool=False, use_kl_penalty: bool=False):
                self.use_distributional = use_distributional
                self.use_delayed = use_delayed
                self.use_magnet = use_magnet
                self.use_kl_penalty = use_kl_penalty
                
class AblationTechniques(Enum):
        CONFIG_0000 = AblationConfig(use_distributional=False, use_delayed=False, use_magnet=False, use_kl_penalty=False)
        CONFIG_0001 = AblationConfig(use_distributional=False, use_delayed=False, use_magnet=False, use_kl_penalty=True)
        CONFIG_0010 = AblationConfig(use_distributional=False, use_delayed=False, use_magnet=True, use_kl_penalty=False)
        CONFIG_0011 = AblationConfig(use_distributional=False, use_delayed=False, use_magnet=True, use_kl_penalty=True)
        CONFIG_0100 = AblationConfig(use_distributional=False, use_delayed=True, use_magnet=False, use_kl_penalty=False)
        CONFIG_0101 = AblationConfig(use_distributional=False, use_delayed=True, use_magnet=False, use_kl_penalty=True)
        CONFIG_0110 = AblationConfig(use_distributional=False, use_delayed=True, use_magnet=True, use_kl_penalty=False)
        CONFIG_0111 = AblationConfig(use_distributional=False, use_delayed=True, use_magnet=True, use_kl_penalty=True)
        CONFIG_1000 = AblationConfig(use_distributional=True, use_delayed=False, use_magnet=False, use_kl_penalty=False)
        CONFIG_1001 = AblationConfig(use_distributional=True, use_delayed=False, use_magnet=False, use_kl_penalty=True)
        CONFIG_1010 = AblationConfig(use_distributional=True, use_delayed=False, use_magnet=True, use_kl_penalty=False)
        CONFIG_1011 = AblationConfig(use_distributional=True, use_delayed=False, use_magnet=True, use_kl_penalty=True)
        CONFIG_1100 = AblationConfig(use_distributional=True, use_delayed=True, use_magnet=False, use_kl_penalty=False)
        CONFIG_1101 = AblationConfig(use_distributional=True, use_delayed=True, use_magnet=False, use_kl_penalty=True)
        CONFIG_1110 = AblationConfig(use_distributional=True, use_delayed=True, use_magnet=True, use_kl_penalty=False)
        CONFIG_1111 = AblationConfig(use_distributional=True, use_delayed=True, use_magnet=True, use_kl_penalty=True)

HYPERPARAMETERS = {
    'Leduc-v0': {
            'lr' : 2.5e-4, # Learning rate of model
            'batch_size' : 64, # Number of time steps that go into the gradient calculation for parameter updating
            'gamma' : 0.99, # Discount factor for future rewards
            'buffer_size' : 10000, # Number of (s,a,r,s,d) tuples remembered
            'epsilon_start' : 1.0, # Starting epsilon/exploration value
            'epsilon_end' : 0.05, # Ending minimum epsilon value
            'epsilon_decay' : 20000, # Number of time steps epsilon decays over
            'V_min' : -5.0, # Minimum atom value in the case of distributional RL
            'V_max' : 5.0, # Maximum atom value
            'num_atoms' : 51, # Number of atoms to be used in distributional RL
            'magnet_scale' : 0.01, # Scale of the magnet loss when using magnet method
            'kl_penalty_scale' : 0.1, # Scale of the kl_penalty loss when using kl_penalty method
            'tau' : 0.005, # Scale of the new network when updating the old network if using a delayed network update
            'hard_update_freq' : 2000, # In the case of NOT using delayed q-network updating, how often is target_q replaced with online_q
    },
    'CartPole-v1': {
            'lr' : 2.5e-4, 
            'batch_size' : 64, 
            'gamma' : 0.99,
            'buffer_size' : 100000,
            'epsilon_start' : 1.0, 
            'epsilon_end' : 0.05,
            'epsilon_decay' : 50000, 
            'V_min' : 0.0, 
            'V_max' : 200.0, 
            'num_atoms' : 51, 
            'magnet_scale' : 0.01,
            'kl_penalty_scale' : 0.1,
            'tau' : 0.005,
            'hard_update_freq' : 2000,
    },
    'MiniGrid-FourRooms-v0': {
            'lr' : 2.5e-4, 
            'batch_size' : 64, 
            'gamma' : 0.99,
            'buffer_size' : 100000,
            'epsilon_start' : 1.0, 
            'epsilon_end' : 0.05,
            'epsilon_decay' : 50000, 
            'V_min' : 0.0, 
            'V_max' : 1.0, 
            'num_atoms' : 51, 
            'magnet_scale' : 0.01,
            'kl_penalty_scale' : 0.1,
            'tau' : 0.005,
            'hard_update_freq' : 2000,
    }
}