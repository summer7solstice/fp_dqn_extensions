import copy
import numpy as np
import torch
import torch.nn.functional as F
from .actions import ActionSelector

def states_preprocessor(states):
    if len(states) == 1:
        return torch.tensor(np.expand_dims(states[0], 0))
    else:
        return torch.tensor(np.array([np.array(s, copy=False) for s in states], copy=False))


def float32_preprocessor(states):
    return torch.tensor(np.array(states, dtype=np.float32))


class TargetNet:
    #  wrapper
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        # sync the targetNet to net
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        target_state = self.target_model.state_dict()
        for x, y in state.items():
            target_state[x] = target_state[x] * alpha + (1 - alpha) * y
        self.target_model.load_state_dict(target_state)


class BaseAgent:
    # the root class
    def init_state(self):
        # at the start, the state is none
        return None


    def __call__(self, states, agent_states):
        # observations and states -> actions
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)
        raise NotImplementedError


class BaseAgentDqn(BaseAgent):
    #  calculates Q values, then sel actions
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    def __call__(self, states, agent_states=None):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        q_values = self.dqn_model(states)
        q = q_values.data.cpu().numpy()
        actions = self.action_selector(q)
        if agent_states is None:
            agent_states = [None] * len(states)
        return actions, agent_states
