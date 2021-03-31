"""
The agent with out brain, do 3 things:
- send the obs to predictor and get the agent
- send the obs to trainer for train model
- send logs to logger
"""

from naive_rllib.models import PPO
from naive_rllib.utils import ZmqAdaptor, get_logger, Monitor
from naive_rllib.configs import get_agent_config, get_zmq_config
import tensorflow as tf
import numpy as np
import copy
import gym

import pickle
import time



class Agent(object):

    def __init__(self, config):

        self.client = ZmqAdaptor(config=config["client"]["sockets"], logger=get_logger())
        print(self.client.sockets)

    @Monitor.dealy()
    def get_action(self, obs):
        self.client.req_predictor.send(pickle.dumps(obs))
        result = self.client.req_predictor.recv()
        result = pickle.loads(result)
        return result['action']

    def send_instance(self):
        """
        1. push to trainer
        2. push to logger
        """

        pass


class AgentWithBrain(object):

    def __init__(self,
                 model=PPO,
                 env="CartPole-v1",
                 lr=0.001,
                 gamma=0.99,
                 lamda=0.95,
                 optimizers='adam',
                 rollout=128,
                 batch_size=256,
                 epoch=4,
                 ppo_eps=0.2,
                 normalize=True,
                 state_size=4,
                 action_size=2,
                 eager=True):
        self.env = gym.make(env)
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.rollout = rollout
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.epoch = epoch
        self.ppo_eps = ppo_eps
        self.normalize = normalize
        self.ppo = model(action_size=self.action_size)
        if optimizers == 'adam':
            self.opt = tf.keras.optimizers.Adam(self.lr)
        else:
            self.opt = tf.keras.optimizers.SGD(self.lr)
        tf.config.run_functions_eagerly(eager)

        self.trainer = ZmqAdaptor(config=get_zmq_config()["trainer"]["sockets"], logger=get_logger())
        print(self.trainer.sockets)

    def get_gaes(self, rewards, dones, values, next_values, gamma, lamda, normalize):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]
        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes, target

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.ppo(state)
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def pub_model(self):
        weights = self.ppo.get_weights()
        start = time.time()
        b_weights = pickle.dumps(weights)
        self.trainer.pub_predictor.send(b_weights)
        print("cost:", time.time()-start)

    def update(self, state, next_state, reward, done, action):
        old_policy, current_value = self.ppo(tf.convert_to_tensor(state, dtype=tf.float32))
        _, next_value = self.ppo(tf.convert_to_tensor(next_state, dtype=tf.float32))
        current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)
        current_value, next_value = np.array(current_value), np.array(next_value)
        old_policy = np.array(old_policy)

        adv, target = self.get_gaes(
            rewards=np.array(reward),
            dones=np.array(done),
            values=current_value,
            next_values=next_value,
            gamma=self.gamma,
            lamda=self.lamda,
            normalize=self.normalize)

        for _ in range(self.epoch):
            sample_range = np.arange(self.rollout)
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]

            batch_state = [state[i] for i in sample_idx]
            batch_done = [done[i] for i in sample_idx]
            batch_action = [action[i] for i in sample_idx]
            batch_target = [target[i] for i in sample_idx]
            batch_adv = [adv[i] for i in sample_idx]
            batch_old_policy = [old_policy[i] for i in sample_idx]

            ppo_variable = self.ppo.trainable_variables

            with tf.GradientTape() as tape:
                tape.watch(ppo_variable)
                train_policy, train_current_value = self.ppo(tf.convert_to_tensor(batch_state, dtype=tf.float32))
                train_current_value = tf.squeeze(train_current_value)
                train_adv = tf.convert_to_tensor(batch_adv, dtype=tf.float32)
                train_target = tf.convert_to_tensor(batch_target, dtype=tf.float32)
                train_action = tf.convert_to_tensor(batch_action, dtype=tf.int32)
                train_old_policy = tf.convert_to_tensor(batch_old_policy, dtype=tf.float32)

                entropy = tf.reduce_mean(-train_policy * tf.math.log(train_policy + 1e-8)) * 0.1
                onehot_action = tf.one_hot(train_action, self.action_size)
                selected_prob = tf.reduce_sum(train_policy * onehot_action, axis=1)
                selected_old_prob = tf.reduce_sum(train_old_policy * onehot_action, axis=1)
                logpi = tf.math.log(selected_prob + 1e-8)
                logoldpi = tf.math.log(selected_old_prob + 1e-8)

                ratio = tf.exp(logpi - logoldpi)
                clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1 - self.ppo_eps,
                                                 clip_value_max=1 + self.ppo_eps)
                minimum = tf.minimum(tf.multiply(train_adv, clipped_ratio), tf.multiply(train_adv, ratio))
                pi_loss = -tf.reduce_mean(minimum) + entropy

                value_loss = tf.reduce_mean(tf.square(train_target - train_current_value))

                total_loss = pi_loss + value_loss

            grads = tape.gradient(total_loss, ppo_variable)
            self.opt.apply_gradients(zip(grads, ppo_variable))

    def learn(self):

        state = self.env.reset()
        episode = 0
        score = 0

        while True:

            state_list, next_state_list = [], []
            reward_list, done_list, action_list = [], [], []

            for _ in range(self.rollout):

                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                score += reward

                if done:
                    if score == 500:
                        reward = 1
                    else:
                        reward = -1
                else:
                    reward = 0

                state_list.append(state)
                next_state_list.append(next_state)
                reward_list.append(reward)
                done_list.append(done)
                action_list.append(action)

                state = next_state

                if done:
                    # print(episode, score)
                    state = self.env.reset()
                    episode += 1
                    score = 0
            self.update(
                state=state_list, next_state=next_state_list,
                reward=reward_list, done=done_list, action=action_list)

            self.pub_model()


if __name__ == '__main__':
    ppo_config = get_agent_config()["ppo"]
    agent = Agent(get_zmq_config())
    # agent.learn()
