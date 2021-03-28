import tensorflow as tf


class PPO(tf.keras.Model):
    def __init__(self, action_size):
        super(PPO, self).__init__()
        self.layer1 = tf.keras.layers.Dense(128, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.layer_a1 = tf.keras.layers.Dense(64, activation='relu')
        self.layer_c1 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(action_size, activation='softmax')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        layer1 = self.layer1(state)
        layer2 = self.layer2(layer1)

        layer_a1 = self.layer_a1(layer2)
        logits = self.logits(layer_a1)

        layer_c1 = self.layer_c1(layer2)
        value = self.value(layer_c1)

        return logits, value
