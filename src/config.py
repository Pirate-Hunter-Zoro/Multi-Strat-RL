import torch

ENV_NAME = 'MiniGrid-FourRooms-v0'
NUM_FRAMES = 50000
HIDDEN_FEATURES = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

class AblationConfig:
        
        def __init__(self, use_distributional: bool=False, use_delayed: bool=False, use_magnet: bool=False, use_kl_penalty: bool=False):
                self.use_distributional = use_distributional
                self.use_delayed = use_delayed
                self.use_magnet = use_magnet
                self.use_kl_penalty = use_kl_penalty

HYPERPARAMETERS = {
    'HalfCheetah-v4': {
            'lr' : 1e-4, # Learning rate of model
            'batch_size' : 64, # Number of time steps that go into the gradient calculation for parameter updating
            'gamma' : 0.9, # Discount factor for future rewards
            'buffer_size' : 100000, # Number of (s,a,r,s,d) tuples remembered
            'epsilon_start' : 1.0, # Starting epsilon/exploration value
            'epsilon_end' : 0.05, # Ending minimum epsilon value
            'epsilon_decay' : 10000, # Number of time steps epsilon decays over
            'techniques' : AblationConfig(use_distributional=True), # The RL technique to use for the environment
            'V_min' : -10, # Minimum atom value in the case of distributional RL
            'V_max' : 10, # Maximum atom value
            'num_atoms' : 51, # Number of atoms to be used in distributional RL
            'magnet_scale' : 0.01, # Scale of the magnet loss when using magnet method
            'kl_penalty_scale' : 1.0, # Scale of the kl_penalty loss when using kl_penalty method
            'tau' : 0.1, # Scale of the new network when updating the old network if using a delayed network update
            'hard_update_freq' : 1000, # In the case of NOT using delayed q-network updating, how often is target_q replaced with online_q
    },
    'MiniGrid-FourRooms-v0': {
            'lr' : 1e-4, 
            'batch_size' : 64, 
            'gamma' : 0.9,
            'buffer_size' : 100000,
            'epsilon_start' : 1.0, 
            'epsilon_end' : 0.05,
            'epsilon_decay' : 10000, 
            'techniques' : AblationConfig(use_delayed=True),
            'V_min' : -10, 
            'V_max' : 10, 
            'num_atoms' : 51, 
            'magnet_scale' : 0.01,
            'kl_penalty_scale' : 1.0,
            'tau' : 0.1,
            'hard_update_freq' : 1000,
    }
}