"""
Contains env and a agent(no brain,only send the obs to predictor and trainer , receive the action predictor
"""
import os
import gym
from naive_rllib.utils import get_logger, package_path
from naive_rllib.agents.ppo import Agent

logger = get_logger(os.path.join(package_path, "logs/client.log"), level="ERROR")


class Instance(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.policys = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, policy, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policys.append(policy)
        self.values.append(value)
        self.dones.append(done)

    def reset(self):
        self.states = [self.states[-1]]
        self.actions = [self.actions[-1]]
        self.rewards = [self.rewards[-1]]
        self.policys = [self.policys[-1]]
        self.values = [self.values[-1]]
        self.dones = [self.dones[-1]]

    def is_full(self):
        return len(self.dones) > self.max_size

    @property
    def dict(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "policys": self.policys,
            "values": self.values,
            "dones": self.dones
        }


class AtariClient(object):

    def __init__(self, env_name, agent, logger, configs=None):
        self.logger = logger
        self.env = self.get_env(env_name)
        self.agent = agent  # (configs["agent"])
        self.rollout = 200
        self.instance = Instance(self.rollout)
        # TODO bind the monitor to the clinet

    def run(self):
        obs = self.env.reset()
        # self.agent.reset()
        episode = 0
        score = 0
        steps = 0
        self.logger.info("Game start {} episode!".format(episode))
        while True:
            while not self.instance.is_full():
                self.env.render()
                # self.logger.debug("waitting get predictor's result")
                action, policy, value = self.agent.get_action(obs)
                # self.logger.debug("get result from predictor success: {}, {}, {}", action, policy, value)
                next_obs, reward, done, info = self.env.step(action)
                score += reward
                steps += 1
                self.instance.add(obs, action, reward, policy, value, done)
                obs = next_obs
                if done:
                    obs = self.env.reset()
                    episode += 1
                    # self.logger.info("Game start {} episode!".format(episode))
                    # self.logger.debug("score{},", score)
                    # push steps, episode, win_num
                    self.agent.push_logger(
                        self.agent.moni.record(score=score, episode=episode, steps=steps).reset())
                    score = 0
            self.agent.push_trainer(self.instance)
            self.instance.reset()

    def get_env(self, env_name):
        env = gym.make(env_name)
        self.logger.info("Create env successfully!")
        return env

if __name__ == '__main__':
    from naive_rllib.configs import get_agent_config, get_zmq_config
    # ppo_config = get_agent_config()["ppo"]
    agent = Agent(get_zmq_config())
    client = AtariClient("CartPole-v1", agent, logger)
    client.run()

