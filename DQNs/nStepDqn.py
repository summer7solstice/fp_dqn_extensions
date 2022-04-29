import gym
import ptan
import argparse
import random

import torch
import torch.optim as optim
from ignite.engine import Engine

import utils
import models
from ptan import baseAgent
import lossCalculator
from epsilonReducer import EpsilonReducer
import ptan.ignite as ptan_ignite
from datetime import timedelta, datetime
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
import torch.optim.lr_scheduler as scheduler

METHOD_NAME = "n_step_dqn"
N_STEPS = 4

# Main change
# Pass the count of steps that we want to unroll on ExperienceSourceFirstLast creation in the steps_count parameter.
# Pass the correct gamma(𝛾**n) to the calc_loss_dqn function.

if __name__ == "__main__":
    result_list = []
    random.seed(utils.RANDOM_SEED)
    torch.manual_seed(utils.RANDOM_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", type=int, default=N_STEPS,
                        help="Steps to do on Bellman unroll")
    args = parser.parse_args()

    game_parameters = utils.PARA_SHORTCUT
    # create the environment and apply a set of standard wrappers
    # render_mode = "human" would show the game screen
    # env = gym.make(game_parameters.environment_name, render_mode = "human")
    env = gym.make(game_parameters.environment_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(utils.RANDOM_SEED)

    # create the NN (double nets)
    device = torch.device("cuda" if args.cuda else "cpu")
    net = models.BasicDQNModel(env.observation_space.shape, env.action_space.n).to(device)
    target_net = baseAgent.TargetNet(net)

    # we create the agent, using an epsilon-greedy action selector as default.
    # During the training, epsilon will be decreased by the EpsilonReducer
    # This will decrease the amount of randomly selected actions and give more control to our NN
    epsilon_reducer = EpsilonReducer()
    agent = baseAgent.BaseAgentDqn(net, device=device, action_selector=epsilon_reducer.action_selector)

    # The next two very important objects are ExperienceSource and ExperienceReplayBuffer.
    # The first one takes the agent and environment and provides transitions over game episodes.
    # Those transitions will be kept in the experience 'replay buffer'.
    experience_source = ptan.experience.ExperienceSourceFirstLast(env=env, agent=agent, gamma=game_parameters.gamma, steps_count=N_STEPS)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(experience_source=experience_source, buffer_size=game_parameters.replay_size)

    # Then we create an optimizer and define the processing function,
    # which will be called for every batch of transitions to train the model.
    # To do this, we call function loss_func of utils and then backpropagate on the result.
    opt = optim.Adam(net.parameters(), lr=game_parameters.lr)

    # scheduler for learning rate decay(gamma is the decay rate), could be used in th future
    # see https://pytorch.org/docs/stable/optim.html
    sched = scheduler.StepLR(opt, step_size=1, gamma=0.1, verbose=True)

    def process_batch(engine, batch):
        opt.zero_grad()
        loss_value = lossCalculator.mse_loss_func(
            batch, net, target_net.target_model,
            gamma=game_parameters.gamma**N_STEPS, device=device)
        loss_value.backward()
        opt.step()
        epsilon_reducer.reduce_by_frames(engine.state.iteration)
        if engine.state.iteration % game_parameters.targetNet_sync_rate == 0:
            #
            target_net.sync()
        return {
            "loss": loss_value.item(),
            "epsilon": epsilon_reducer.action_selector.epsilon,
        }

    # And, finally, we create the Ignite Engine object, configure it using a function from
    # common.py, and run our training process.
    # engine = Engine(process_batch)
    # utils.setup_ignite(engine, game_parameters, experience_source, METHOD_NAME, epsilon_reducer)
    # engine.run(utils.create_batch(replay_buffer))
    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(experience_source, bound_avg_reward=game_parameters.goal_reward).attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)


    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('fps', 0),
            timedelta(seconds=trainer.state.metrics.get('time_passed', 0))))
        # if trainer.state.episode % 2 == 0:
        #     sched.step()
        #     print("LR decrease to", sched.get_last_lr()[0])
        result_list.append((trainer.state.episode, trainer.state.episode_reward))

    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics['time_passed']),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True
        print("solved epoch %d", trainer.state.epoch)
        print("--------Finished---------")
        print(result_list)
        for obj in result_list:
            print(obj)

    # track TensorBoard data
    logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{game_parameters.game_name}-{METHOD_NAME}={METHOD_NAME}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=['reward', 'steps', 'avg_reward'])
    tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss', 'avg_fps'],
                                      output_transform=lambda a: a)
    tb.attach(engine, log_handler=handler, event_name=ptan_ignite.PeriodEvents.ITERS_100_COMPLETED)

    engine.run(utils.create_batch(replay_buffer))

