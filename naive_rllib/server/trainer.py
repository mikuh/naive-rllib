import pickle
import numpy as np
import tensorflow as tf
from naive_rllib.utils import ZmqAdaptor, get_logger
from naive_rllib.configs import get_zmq_config
from naive_rllib.models.ppo import PPO


class Trainer(object):

    def __init__(self):
        self.action_size = 2
        self.ppo = PPO(action_size=2)
        self.trainer = ZmqAdaptor(config=get_zmq_config()["trainer"]["sockets"], logger=get_logger())
        self.epoch = 4
        self.batch_size = 128
        self.ppo_eps = 0.2
        self.opt = tf.keras.optimizers.Adam(0.001)

        self.ppo(np.array([[0.0, 0.0,0.0, 0.0]]))

    def update(self):
        data = self.trainer.pull_client.recv()
        data = pickle.loads(data)
        for _ in range(self.epoch):
            sample_range = np.arange(len(data.dones))
            np.random.shuffle(sample_range)
            sample_idx = sample_range[:self.batch_size]

            batch_state = [data.state[i] for i in sample_idx]
            # batch_done = [done[i] for i in sample_idx]
            batch_action = [data.action[i] for i in sample_idx]
            batch_target = [data.target[i] for i in sample_idx]
            batch_adv = [data.adv[i] for i in sample_idx]
            batch_old_policy = [data.policy[i] for i in sample_idx]

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

    def pub_model(self):

        # print(self.ppo.get_weights())
        self.trainer.pub_predictor.send(pickle.dumps(self.ppo.get_weights()))

    def run(self):
        n = 0
        while True:
            n += 1
            self.pub_model()
            print("pub model success")
            self.update()
            # break


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
