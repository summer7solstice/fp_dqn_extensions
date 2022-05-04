import numpy as np
import torch
from types import SimpleNamespace

import ptan


RANDOM_SEED = 123

GAME_PARAMETERS = {
    'pong' : SimpleNamespace(**
        {
            'environment_name':     "PongNoFrameskip-v4",
            'goal_reward':          18.0,
            'game_name':            'pong',
            'replay_size':          10**5,
            'replay_start':         10**4,
            'targetNet_sync_rate':  10**3,
            'epsilon_frames':       10**5,
            'epsilon_start':        1.0,
            'epsilon_final':        0.02,
            'lr':                   0.0001,
            'gamma':                0.99,
            'batch_size':           32
        })
}
PARA_SHORTCUT = GAME_PARAMETERS['pong']


# func to calculate state_values
@torch.no_grad()
def calculate_state_values(states, net, device="cpu"):
    values = []
    for batch in np.array_split(states, 64):
        states_values = torch.tensor(batch).to(device)
        action_values = net(states_values)
        best_action_values_v = action_values.max(1)[0]
        values.append(best_action_values_v.mean().item())
    return np.mean(values)


# more evaluation controlled by proper frequency
@torch.no_grad()
def evaluate_states(states, net, device, engine):
    s_v = torch.tensor(states).to(device)
    adv, val = net.adv_val(s_v)
    engine.state.metrics['adv'] = adv.mean().item()
    engine.state.metrics['val'] = val.mean().item()

# takes ExperienceReplayBuffer
# and infinitely generates training batches sampled from the buffer.
# In the beginning, the function ensures that the buffer contains the required amount of samples.
def create_batch(buffer: ptan.experience.ExperienceReplayBuffer):
    step_size = 1
    buffer.populate(PARA_SHORTCUT.replay_start)
    while 1:
        buffer.populate(step_size)
        yield buffer.sample(PARA_SHORTCUT.batch_size)


# it takes the batch of transitions and converts it into the set of NumPy arrays suitable for training
def get_batch(batch:[ptan.experience.ExperienceFirstLast]):
    states = []
    actions = []
    rewards = []
    done = []
    last_state = []
    for index in range(len(batch)):
        experience = batch[index]
        state = np.array(experience.state)
        states.append(state)
        actions.append(experience.action)
        rewards.append(experience.reward)
        done.append(experience.last_state is None)
        if experience.last_state is not None:
            last_state.append(np.array(experience.last_state))
        else:
            last_state.append(state)
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(done, dtype=np.uint8), np.array(last_state, copy=False)