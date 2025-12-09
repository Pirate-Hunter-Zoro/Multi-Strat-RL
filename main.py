from src.agent import RainbowAgent
from src.buffers import ReplayBuffer
from src.config import ENV_NAME, NUM_FRAMES, HYPERPARAMETERS, DEVICE
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import torch
import minigrid # Simply importing will register the environment in the background

def main():
    config = HYPERPARAMETERS[ENV_NAME]
    batch_size = config['batch_size']
    buffer_size = config['buffer_size']
    ablation_config = config['techniques']
    tau = config['tau']
    hard_update_freq = config['hard_update_freq']
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay = config['epsilon_decay']
    
    env = gym.make(id=ENV_NAME, render_mode="rgb_array")
    env = FlattenObservation(env)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n # TODO - my interpretter is not flagging '.n' as an attribute - pretty sure it's wrong
    
    agent = RainbowAgent(state_dim=state_dim, num_actions=n_actions, config_dict=config)
    buf = ReplayBuffer(state_dim=state_dim, max_size=buffer_size)
    rewards_per_episode = []
    curr_episode_reward = 0.0
    curr_epsilon = epsilon_start
    decay_step = (epsilon_start - epsilon_end) / epsilon_decay
    curr_state, _ = env.reset()
    for i in range(NUM_FRAMES):
        # Take an action based on the state and add the experience to our buffer
        curr_state = torch.tensor(curr_state, dtype=torch.float32).unsqueeze(0).to(DEVICE) # batch of size 1
        action = agent.select_action(state=curr_state, epsilon=curr_epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        curr_episode_reward += reward
        done = terminated or truncated
        buf.add(state=curr_state.cpu().numpy()[0], action=action, reward=reward, next_state=next_state, done=done)
        if done:
            curr_state, _ = env.reset()
            rewards_per_episode.append(curr_episode_reward)
            curr_episode_reward = 0
            print(f"Episode: {len(rewards_per_episode)} | Reward: {curr_episode_reward:.2f} | Epsilon: {curr_epsilon:.3f}               ", end='\r')
        else:
            curr_state = next_state
            
        if buf.size >= batch_size:
            # We can use the buffer to learn
            batch = buf.sample(batch_size=batch_size)
            agent.update_model(batch=batch)
        
        # Update epsilon to decay linearly over the desired number of steps down to epsilon_end
        curr_epsilon = max(epsilon_end, curr_epsilon - decay_step)
        # Update target network
        if ablation_config.use_delayed or (i % hard_update_freq == 0):
            agent.target_update(tau=tau, use_delayed=ablation_config.use_delayed)
        
            
if __name__ == "__main__":
    main()