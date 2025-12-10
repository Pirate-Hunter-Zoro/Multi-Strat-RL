import rlcard
import random
import numpy as np

class _DummyActionSpace:
        def __init__(self, num_actions: int):
            self.n = num_actions
    
class _DummyObservationSpace:
    def __init__(self, state_shape: any):
        self.shape = state_shape

class RLCardWrapper:
    
    def __init__(self):
        self.leduc_env = rlcard.make('leduc-holdem')
        self.action_space = _DummyActionSpace(self.leduc_env.num_actions)
        self.observation_space = _DummyObservationSpace(self.leduc_env.state_shape[0])
    
    def reset(self) -> tuple[any, dict]:
        """
        Reset wrapper - the environment reset should give player id 0 which is what we want
        
        :return: Initial State and info
        :rtype: Any
        """
        state, _ = self.leduc_env.reset()
        return state['obs'], {}
    
    def step(self, action: int) -> tuple[any, float, bool, bool, dict]:
        """
        Play move and ensure that opponent goes before returning state
        
        :param action: Action agent takes
        :type action: int
        :return: next state, reward, terminated, truncated, info
        :rtype: tuple[Any, float, bool, bool, dict]
        """
        next_state, next_player_id = self.leduc_env.step(action)
        while (not self.leduc_env.is_over()) and (not next_player_id == 0):
            next_action = random.randint(0, self.action_space.n - 1)
            next_state, next_player_id = self.leduc_env.step(next_action)
        if self.leduc_env.is_over():
            reward = self.leduc_env.get_payoffs()[0]
        else:
            reward = 0
        next_state = np.array(next_state['obs'], dtype=np.float32)
        return (next_state, reward, self.leduc_env.is_over(), self.leduc_env.is_over(), {})