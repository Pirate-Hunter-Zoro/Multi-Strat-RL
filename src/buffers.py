import torch


class ReplayBuffer:
    
    def __init__(self, state_dim: int, max_size: int):
        self.ptr = 0
        self.size = 0
        self.max_size = max_size
        self.state = torch.zeros(size=(max_size, state_dim))
        self.action = torch.zeros(size=(max_size, 1), dtype=torch.int64)
        self.reward = torch.zeros(size=(max_size, 1))
        self.next_state = torch.zeros_like(self.state)
        self.done = torch.zeros(size=(max_size, 1))
    
    def add(self, state: torch.tensor, action: int, reward: float, next_state: torch.tensor, done: int):
        """Adds experience to the buffer while maintaining a circular queue

        Args:
            state (torch.tensor): State in the experience
            action (int): Action taken
            reward (float): Observed reward
            next_state (torch.tensor): Resulting next state
            done (int): Flag (0/1) to indicate if episode is done after this experience
        """
        self.state[self.ptr] = torch.tensor(state, dtype=torch.float32)
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = torch.tensor(next_state)
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> tuple[torch.tensor]:
        indices = torch.randint(high=self.size, size=(batch_size,))
        return (self.state[indices], self.action[indices], self.reward[indices], self.next_state[indices], self.done[indices])