from src.agent import RainbowAgent
from src.buffers import ReplayBuffer
from src.config import ENV_NAME, HYPERPARAMETERS

def main():
    config = HYPERPARAMETERS[ENV_NAME]
    lr = config['lr']
    batch_size = config['batch_size']
    gamma = config['gamma']
    buffer_size = config['buffer_size']
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay = config['epsilon_decay']
    ablation_config = config['techniques']
    V_min = config['V_min']
    V_max = config['V_max']
    num_atoms = config['num_atoms']
    
            
if __name__ == "__main__":
    main()