import numpy as np
import torch
from types import SimpleNamespace

import ptan
import epsilonReducer
from ignite.engine import Engine
import modifiedIgnite as ptan_ignite
import warnings
from datetime import timedelta, datetime
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from typing import Iterable, List
RANDOM_SEED = 123

GAME_PARAMETERS = {
    'pong' : SimpleNamespace(**
        {
            'environment_name':     "PongNoFrameskip-v4",
            'goal_reward':          19.0,
            'game_name':            'pong',
            'replay_size':          1000000,
            'replay_start':         10000,
            'targetNet_sync_rate':  1000,
            'epsilon_frames':       1000000,
            'epsilon_start':        1.0,
            'epsilon_final':        0.1,
            'lr':                   0.00025,
            'gamma':                0.99,
            'batch_size':           32
        })
}
PARA_SHORTCUT = GAME_PARAMETERS['pong']


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
def get_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states = []
    actions = []
    rewards = []
    done = []
    last_state = []
    for experience in batch:
        # experience = batch[index]
        state = np.array(experience.state, copy=False)
        states.append(state)
        actions.append(experience.action)
        rewards.append(experience.reward)
        done.append(experience.last_state is None)
        # if experience.last_state is not None:
        #     last_state.append(np.array(experience.last_state))
        # else:
        #     last_state.append(state)
        if experience.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(experience.last_state, copy=False)
        last_state.append(lstate)
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(done, dtype=np.uint8), np.array(last_state, copy=False)


# attaches the needed Ignite handlers, showing the training progress and writing metrics to TensorBoard.
def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str, epsilon_tracker: epsilonReducer,
                 extra_metrics: Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.goal_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

        # if trainer.state.episode % 5 == 0:
        #     # print(exp_source.agent.dqn_model.state_dict())
        #     print(epsilon_tracker.selector.epsilon)
        #     print(run_name)

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    # The rest of the function is related to the TensorBoard data that we want to track:
    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs/{now}-{params.game_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    # Our processing function will return the loss value, so we attach the
    # RunningAverage transformation (also provided by Ignite) to get a smoothed version of the loss over time
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    # metrics (calculated during the training and kept in the engine state)
    metrics = ['reward', 'steps', 'avg_reward']
    # metrics are updated at the end of every game episode
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)