from collections import deque
import numpy as np
from tqdm.auto import trange, tqdm
import torch


def train_dqn(env, agent, brain_name="BananaBrain", n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=.01, eps_decay=.995):
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
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores