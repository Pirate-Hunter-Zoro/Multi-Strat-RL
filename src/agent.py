import torch
import torch.optim as optim
import torch.nn.functional as f
from src.networks import RainbowDQN
import copy
import random

class RainbowAgent:
    
    def __init__(self, state_dim: int, num_actions: int, config_dict: dict):
        self.gamma, self.v_min, self.v_max, self.num_atoms = config_dict['gamma'], config_dict['V_min'], config_dict['V_max'], config_dict['num_atoms']
        self.online_net = RainbowDQN(state_dim=state_dim, num_actions=num_actions, num_atoms=self.num_atoms, config=config_dict['techniques'])
        self.target_net = copy.deepcopy(self.online_net)
        lr = config_dict['lr']
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.num_actions = num_actions
        self.support = None
        if config_dict['techniques'].use_distributional:
            self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
    
    def select_action(self, state: torch.tensor, epsilon: float) -> int:
        """Given one state, select an action via an epsilon-greedy policy

        Args:
            state (torch.tensor): Individual vectorized state
            epsilon (float): Probability of taking a random action

        Returns:
            int: Resulting action
        """
        self.support = self.support.to(state.device)
        if random.random() < epsilon:
            # Random action
            return int(random.random() * self.num_actions)
            
        if self.support != None:
            # Using distributional RL
            logits = self.online_net(state) # 1 x NUM_ACTIONS x NUM_ATOMS
            probs = torch.softmax(input=logits, dim=-1)
            expected_rewards = probs * (self.support.reshape(1, 1, len(self.support)))
            expected_rewards = torch.sum(expected_rewards, dim=-1)
            return torch.argmax(input=expected_rewards, dim=-1)
        else:
            # Non-distributional RL
            return torch.argmax(self.online_net(state), dim=-1)
        
    def update_model(self, batch: torch.tensor):
        """Update the online and target networks based on the batch of observations

        Args:
            batch (torch.tensor): states, actions, rewards, next states, dones
        """
        s, a, r, next_s, d = batch[0], batch[1], batch[2], batch[3], batch[4]
        if self.support != None: # C51
            # Estimate atom probabilities given the experience - states and actions are known at each step
            atoms = self.num_atoms * torch.arange(start=0, end=batch.shape[0], step=1) # [0, num_atoms, 2*num_atoms, etc...]
            atoms = atoms.reshape(size=(batch.shape[0],1))
            
            # Use the online network to select the best next actions
            next_atom_logits = self.online_net(next_s)
            next_atom_probs = torch.softmax(next_atom_logits)
            weighted_next_atom_vals = next_atom_probs * self.support.reshape(1, 1, len(self.support))
            expected_next_atom_vals = torch.sum(input=weighted_next_atom_vals, dim=-1)
            next_actions = torch.argmax(input=expected_next_atom_vals, dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1,self.num_atoms) # BATCH x 1 x NUM_ATOMS
            
            # Use the target network to find the atom probabilities associated with the action selected by the online network
            next_atom_logits_target = self.target_net(next_s)
            next_atom_probs_target = torch.softmax(input=next_atom_logits_target, dim=-1)
            next_atom_probs_for_taken_actions = torch.gather(next_atom_probs_target, dim=1, index=next_actions).squeeze() # BATCH_SIZE x NUM_ATOMS
            
            # Project the atom probabilities
            delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            # Given our new rewards, we need to shift the atom probabilities
            new_atom_rewards = r + self.gamma * self.support * (1 - d) # BATCH_SIZE x NUM_ATOMS
            new_atom_rewards = torch.clamp(input=new_atom_rewards, min=self.v_min, max=self.v_max)
            # Over all the batches, for each 'atom reward', find the bin it falls under
            bins = (new_atom_rewards - self.v_min) / delta_z # BATCH_SIZE x NUM_ATOMS
            l = torch.floor(input=bins) # BATCH_SIZE x NUM_ATOMS
            u = torch.ceil(input=bins) # BATCH_SIZE x NUM_ATOMS
            l, u = torch.clamp(input=l, min=0, max=self.num_atoms), torch.clamp(input=u, min=0, max=self.num_atoms)
            upper_weight = bins - l # BATCH_SIZE x NUM_ATOMS
            lower_weight = u - bins # BATCH_SIZE x NUM_ATOMS
            l = l + atoms # batch zero writes to indices 0-num_atons-1, batch 1 to num_atoms-2*num_atoms-1, etc.
            u = u + atoms
            upper_bin_weighted_prob_addition = upper_weight * next_atom_probs_for_taken_actions # BATCH_SIZE x NUM_ATOMS
            lower_bin_weighted_prob_addition = lower_weight * next_atom_probs_for_taken_actions # BATCH_SIZE x NUM_ATOMS
           
            # Update these two bin probabilities, over all the batches
            m = torch.zeros(size=(batch.shape[0], self.num_atoms)) # BATCH X NUM_ATOMS
            # NOTE - the index_add_ - with the trailing '_' in the function call - is an in-place operation
            m.view(-1).index_add_(dim=0, index=u, source=upper_bin_weighted_prob_addition.flatten()) # Over all the batches, accumulate the weights given to each 'atom bin'
            m.view(-1).index_add_(dim=0, index=l, source=lower_bin_weighted_prob_addition.flatten())
            
            # Now we calculate the loss since m gives us out target atom probabilities
            atom_logits = self.online_net(s)
            atom_log_softmax = torch.log_softmax(input=atom_logits, dim=-1)
            # We only care about the log-softmax atom probabilities associated with the actions our agent took
            action_indices = a.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1,self.num_atoms) # BATCH x 1 x NUM_ATOMS
            atom_log_softmax_for_taken_actions = torch.gather(input=atom_log_softmax, dim=1, index=action_indices).squeeze() # BATCH x NUM_ATOMS
            
            # We now have the target atom probabilities over all batches
            loss = -torch.sum(input=m * atom_log_softmax_for_taken_actions, dim=1) / batch.shape[0]
        else:
            q_values_for_all_actions = self.online_net(s)
            q_values = torch.gather(input=q_values_for_all_actions, dim=1, index=a)
            # Achieve the max q-values of the next state according to the target network
            next_q_values_for_all_actions = self.target_net(next_s) # BATCH x NUM_ACTIONS
            next_actions = torch.argmax(input=next_q_values_for_all_actions, dim=1).unsqueeze(1) # BATCH x 1
            next_q_values = torch.gather(input=next_q_values_for_all_actions, dim=1, index=next_actions)
            target_q_values = (r + self.gamma * next_q_values * (1 - d)).detach()
            loss = f.mse_loss(input=q_values, target=target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def target_update(self, tau: float, use_delayed: bool=True):
        """Update the target network based on the 'tau' parameter and the online network (which is trained via batches)

        Args:
            tau (float): Weight of the online network in the updating of the target network
            use_delayed (bool): Flag for if the target network is completely replaced with the online network or only through a delayed weighting
        """
        tau_value = tau if use_delayed else 1.0
        for online_param, target_param in zip(self.online_net.parameters(), self.target_net.parameters()):
            new_v = tau_value * online_param.data + (1-tau_value) * target_param.data
            target_param.data.copy_(other=new_v)