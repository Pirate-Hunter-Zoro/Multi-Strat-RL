ENV_NAME = 'MiniGrid-FourRooms-v0'

RL_TECHNIQUE_OPTIONS = ['magnet', 'distributional', 'delayed']

HYPERPARAMETERS = {
    'HalfCheeta-v4': {
            'lr' : 1e-4, # Learning rate of model
            'batch_size' : 64, # Number of time steps that go into the gradient calculation for parameter updating
            'gamma' : 0.9, # Discount factor for future rewards
            'buffer_size' : 1000, # Number of (s,a,r,s,d) tuples remembered
            'epsilon_start' : 1.0, # Starting epsilon/exploration value
            'epsilon_end' : 0.05, # Ending minimum epsilon value
            'epsilon_decay' : 10000, # Number of time steps epsilon decays over
            'technique' : 'distributional', # The RL technique to use for the environment
    },
    'MiniGrid-FourRooms-v0': {
            'lr' : 1e-4, 
            'batch_size' : 64, 
            'gamma' : 0.9,
            'buffer_size' : 1000,
            'epsilon_start' : 1.0, 
            'epsilon_end' : 0.05,
            'epsilon_decay' : 10000, 
            'technique' : 'delayed',
    }
}