from collections import deque
import numpy as np
from tqdm.notebook import trange, tqdm
import torch


def train_dqn(env, agent, brain_name="BananaBrain", n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=.01,
              eps_decay=.995, termination_threshold=13):
    """
    Initiates a training sequence of the model of given agent. Training will continue for given number of episodes and
    each episode will take maximum max_t time steps. The policy will be epsilon greedy and epsilon will decay for a
    given decay coefficient. The training will be terminated if average score of 100 episodes exceed the given
    termination threshold. A checkpoint will be (checkpoint.pth) created/updated each time when a better score is
    obtained or at the end of the training.
    :param env: Unity environment
    :param agent: Created agent instance
    :param brain_name: Name of the brain respect to given environment
    :param n_episodes: Number of maximum episodes
    :param max_t: Maximum time steps per episde
    :param eps_start: Starting value of the epsilon value
    :param eps_end: Ending value of the epsilon value
    :param eps_decay: Decaying factor for the epsilon value
    :param termination_threshold: Termination threshold
    :return: Scores over episodes
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    tqdm_bar = trange(1, n_episodes, desc="Episode")
    inner_bar = tqdm(total=max_t)
    for i_episode in tqdm_bar:
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        for t in range(1, max_t):
            action = agent.act(state, eps)

            env_info = env.step(action.astype(int))[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            inner_bar.set_description("Time Step T: {}, Score: {}".format(t, score))
            inner_bar.update()
            if done:
                break
        inner_bar.reset()
        tqdm_bar.set_description("Episode: {}, Score: {}".format(i_episode, score))
        scores_window.append(score)       # save most recent score
        try:
            if scores_window[-1] > scores_window[-2]:
                agent.save_model()
        except IndexError:
            pass

        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        if np.mean(scores_window)>termination_threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            agent.save_model('checkpoint.pth')
            break
    return scores