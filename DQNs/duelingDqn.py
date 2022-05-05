import gym
import ptan
import argparse
import random
import torch
import torch.optim as optim
from ignite.engine import Engine
import numpy as np
import utils
import models
from ptan import baseAgent
import ptan.ignite as ptan_ignite
import lossCalculator
from epsilonReducer import EpsilonReducer
from datetime import timedelta, datetime
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
import warnings
METHOD_NAME = "dueling_dqn"
BUFFER_EVALUATE_SIZE = 1000
EVALUATE_FRE_BY_FRAME = 100

if __name__ == "__main__":
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)
    random.seed(utils.RANDOM_SEED)
    torch.manual_seed(utils.RANDOM_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    game_parameters = utils.PARA_SHORTCUT
    # create the environment and apply a set of standard wrappers
    # render_mode = "human" would show the game screen
    env = gym.make(game_parameters.environment_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(utils.RANDOM_SEED)

    # create the NN (double nets)
    device = torch.device("cuda" if args.cuda else "cpu")
    net = models.DuelingDQNModel(env.observation_space.shape, env.action_space.n).to(device)

    target_net = baseAgent.TargetNet(net)

    # we create the agent, using an epsilon-greedy action selector as default.
    # During the training, epsilon will be decreased by the EpsilonReducer
    # This will decrease the amount of randomly selected actions and give more control to our NN
    epsilon_reducer = EpsilonReducer()
    agent = baseAgent.BaseAgentDqn(net, device=device, action_selector=epsilon_reducer.action_selector)

    # The next two very important objects are ExperienceSource and ExperienceReplayBuffer.
    # The first one takes the agent and environment and provides transitions over game episodes.
    # Those transitions will be kept in the experience 'replay buffer'.
    experience_source = ptan.experience.ExperienceSourceFirstLast(env=env, agent=agent, gamma=game_parameters.gamma)
    replay_buffer = ptan.experience.ExperienceReplayBuffer(experience_source=experience_source, buffer_size=game_parameters.replay_size)

    # Then we create an optimizer and define the processing function,
    # which will be called for every batch of transitions to train the model.
    # To do this, we call function loss_func of utils and then backpropagate on the result.
    opt = optim.Adam(net.parameters(), lr=game_parameters.lr)

    # scheduler for learning rate decay(gamma is the decay rate), could be used in th future
    # see https://pytorch.org/docs/stable/optim.html
    # sched = scheduler.StepLR(opt, step_size=1, gamma=0.1, verbose=True)

    def process_batch(engine, batch):
        opt.zero_grad()
        loss_value = lossCalculator.mse_loss_func(
            batch, net, target_net.target_model,
            gamma=game_parameters.gamma, device=device)
        loss_value.backward()
        opt.step()
        epsilon_reducer.reduce_by_frames(engine.state.iteration)
        if engine.state.iteration % game_parameters.targetNet_sync_rate == 0:
            # sync the net
            target_net.sync()
        if engine.state.iteration % EVALUATE_FRE_BY_FRAME == 0:
            eval_states = getattr(engine.state, "eval_states", None)
            if eval_states is None:
                eval_states = replay_buffer.sample(BUFFER_EVALUATE_SIZE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)
                engine.state.eval_states = eval_states
            utils.evaluate_states(eval_states, net, device, engine)
        return {
            "loss": loss_value.item(),
            "epsilon": epsilon_reducer.action_selector.epsilon,
        }

    # finally, we create the Ignite Engine object
    # engine = Engine(process_batch)
    # utils.setup_ignite(engine, game_parameters, experience_source, METHOD_NAME, epsilon_reducer)
    # engine.run(utils.create_batch(replay_buffer))
    engine = Engine(process_batch)
    ptan_ignite.EndOfEpisodeHandler(experience_source, bound_avg_reward=game_parameters.goal_reward).attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)


    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        print("Episode %d: reward=%s, steps=%s, speed=%.3f frames/s, elapsed=%s, loss=%lf" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps, trainer.state.metrics.get('fps', 0),
            timedelta(seconds=trainer.state.metrics.get('time_passed', 0)),
            trainer.state.output["loss"]
        ))
        # if trainer.state.episode % 2 == 0:
        #     sched.step()
        #     print("LR decrease to", sched.get_last_lr()[0])


    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=trainer.state.metrics['time_passed']),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True
        print("solved epoch %d", trainer.state.epoch)


    # track TensorBoard data
    logdir = f"runs/{datetime.now().isoformat(timespec='minutes')}-{game_parameters.game_name}-{METHOD_NAME}={METHOD_NAME}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    episode_handler = tb_logger.OutputHandler(tag="episodes", metric_names=metrics)
    tb.attach(engine, log_handler=episode_handler, event_name=ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)

    # write to tensorboard every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(('adv', 'val'))
    handler = tb_logger.OutputHandler(tag="train", metric_names=metrics,
                                      output_transform=lambda a: a)
    tb.attach(engine, log_handler=handler, event_name=ptan_ignite.PeriodEvents.ITERS_100_COMPLETED)

    engine.run(utils.create_batch(replay_buffer))


