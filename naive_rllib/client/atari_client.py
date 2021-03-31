"""
Contains env and a agent(no brain,only send the obs to predictor and trainer , receive the action predictor
"""
import os
import gym
from naive_rllib.utils import get_logger, package_path

logger = get_logger(os.path.join(package_path, "logs/client.log"))


# class Instance(object):
#     def __init__(self, state=None, state_value=0, action=None, action_prob=None, q_reward=0, gae_advantage=0,
#                  action_mask=None, instant_reward=0, lstm_state=None, lstm_mask=None):
#         self.state = state
#         self.state_value = state_value
#         self.action = action
#         self.action_prob = action_prob
#         self.q_reward = q_reward
#         self.gae_advantage = gae_advantage
#         self.action_mask = action_mask
#         self.lstm_state = lstm_state
#         self.lstm_mask = lstm_mask
#         self.instant_reward = instant_reward

class Instance(object):

    def __init__(self, state=None, action=None, reward=None, next_state=None, **kwargs):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

        for k, v in kwargs.items():
            self.__setattr__(k, v)


class AtariClient(object):

    def __init__(self, env_name, agent, logger, configs):
        self.env = self.get_env(env_name)
        self.agent = agent(configs["agent"])
        self.logger = logger
        self.rollout = 200

    def run(self):
        obs = self.env.reset()
        self.agent.reset()
        episode = 0
        score = 0
        self.logger.info("Game start {} episode!".format(episode))
        while True:
            instances = []
            for _ in range(self.rollout):
                action = self.agent.get_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                score += reward
                instances.append(
                    Instance(state=obs, action=action, reward=reward, next_state=next_obs, done=done, info=info))
                obs = next_obs
                if done:
                    obs = self.env.reset()
                    episode += 1
                    score = 0
                    self.logger.info("Game start {} episode!".format(episode))

            self.agent.push_trainer(instances)

    def get_env(self, env_name):
        env = gym.make(env_name)
        return env
