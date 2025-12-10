from src.agent import RainbowAgent
from src.buffers import ReplayBuffer
from src.config import NUM_FRAMES, HYPERPARAMETERS, DEVICE, AblationTechniques
from src.wrappers import RLCardWrapper
import gymnasium as gym
import torch
from minigrid.wrappers import FlatObsWrapper
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

def run_ablation(env_name: str, ablation_technique_set: AblationTechniques) -> tuple[list[float], pd.Series]:
    """Run reinforcement learning techniques on the given environment specified by the given ablation techniques

    Args:
        env_name (str): Environment to run RL on
        ablation_technique_set (AblationTechniques): Contains information on the ablation techniques to use

    Returns:
        tuple[list[float], pd.Series]: Rewards and smoothed rewards by the episode
    """
    config = HYPERPARAMETERS[env_name]
    batch_size = config['batch_size']
    buffer_size = config['buffer_size']
    tau = config['tau']
    hard_update_freq = config['hard_update_freq']
    epsilon_start = config['epsilon_start']
    epsilon_end = config['epsilon_end']
    epsilon_decay = config['epsilon_decay']
    
    if "Leduc" in env_name:
        env = RLCardWrapper()
    else:
        env = gym.make(id=env_name, render_mode="rgb_array")
        if "MiniGrid" in env_name:
            env = FlatObsWrapper(env)
    
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[0] # Continuous action space case
    
    agent = RainbowAgent(state_dim=state_dim, num_actions=n_actions, config_dict=config, ablation_config=ablation_technique_set)
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
            print(f"Episode: {len(rewards_per_episode)} | Reward: {curr_episode_reward:.2f} | Epsilon: {curr_epsilon:.3f} | Percent Through Training: {i*100/NUM_FRAMES:.3f}%              ", end='\r')
            curr_episode_reward = 0
        else:
            curr_state = next_state
            
        if buf.size >= batch_size:
            # We can use the buffer to learn
            batch = buf.sample(batch_size=batch_size)
            agent.update_model(batch=batch)
        
        # Update epsilon to decay linearly over the desired number of steps down to epsilon_end
        curr_epsilon = max(epsilon_end, curr_epsilon - decay_step)
        # Update target network
        if ablation_technique_set.value.use_delayed or (i % hard_update_freq == 0):
            agent.target_update(tau=tau, use_delayed=ablation_technique_set.value.use_delayed)
    
    # Now that training is done, plot the results
    reward_series = pd.Series(data=rewards_per_episode)
    smoothed_rewards = reward_series.ewm(span=20).mean() # Assigns higher weight to more recent rewards
    
    return rewards_per_episode, smoothed_rewards

def obtain_results(env_name: str):
    """Obtain results for all ablation techniques on the given environment

    Args:
        env_name (str): Environment to run RL on
    """
    # Create a plot for all ablation techniques
    plt.figure(figsize=(10, 5))
    for technique in AblationTechniques:
        print(f"Running technique: {technique.name} with config: Distributional={technique.value.use_distributional}, Delayed={technique.value.use_delayed}, Magnet={technique.value.use_magnet}, KL Penalty={technique.value.use_kl_penalty}")
        rewards_per_episode, smoothed_rewards = run_ablation(env_name=env_name, ablation_technique_set=technique)
        # Save results
        os.makedirs(Path(f"results/{env_name}/{technique.name}"), exist_ok=True)
        pd.Series(data=rewards_per_episode).to_csv(Path(f"results/{env_name}/{technique.name}/raw_rewards.csv"), index=False)
        smoothed_rewards.to_csv(Path(f"results/{env_name}/{technique.name}/smoothed_rewards.csv"), index=False)
        
        # Add this ablation technique's results to the graph
        plt.plot(rewards_per_episode, label=None, alpha=0.3) 
        # Only label the smoothed line reward - this will cut our number of labels in half and increase readability
        plt.plot(smoothed_rewards, label=f'EMA - {technique.name}', alpha=1.0)
    
    plt.legend()
    plt.title(f'Raw and Smoothed Rewards by the Episode: {env_name}')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(f"results/{env_name}_all_ablation_techniques.png")
    plt.close()

def main():
    for env in HYPERPARAMETERS.keys():
        obtain_results(env_name=env)    
            
if __name__ == "__main__":
    main()