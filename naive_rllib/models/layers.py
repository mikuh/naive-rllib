import tensorflow as tf


class DenseEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs):
        output = self.layer1(inputs)
        output = self.layer2(output)
        return output
