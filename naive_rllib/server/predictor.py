from naive_rllib.utils import ZmqAdaptor, get_logger
from naive_rllib.configs import get_zmq_config
from naive_rllib.models.ppo import PPO
import pickle
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# tf.compat.v1.disable_eager_execution()

class Predictor(object):

    def __init__(self):
        self.action_size = 2
        self.model = PPO(action_size=2)
        self.predictor = ZmqAdaptor(config=get_zmq_config()["predictor"]["sockets"], logger=get_logger())
        self.data = None
        self.model(np.array([[0.0, 0.0, 0.0, 0.0]]))

    # @tf.function
    def predict(self):
        """get the model result
        """
        data = self.predictor.router_client.recv_multipart()
        obs = pickle.loads(data[-1])
        print("receive data:", obs)
        policy, value = self.model(np.array([obs]))
        print(policy, value)
        policy = np.array(policy)[0]
        action = np.random.choice(self.action_size, p=policy)
        send_data = {"action": action, "policy": policy, "value": value.numpy()[0][0]}
        data[-1] = pickle.dumps(send_data)
        self.predictor.router_client.send_multipart(data)
        print("send data:", send_data)

    def sub_model(self):
        weights = self.predictor.sub_trainer.recv()
        weights = pickle.loads(weights)
        self.model.set_weights(weights)

    def push_logger(self):
        pass

    def run(self):
        while True:
            self.sub_model()
            print("sub success")
            self.predict()



# predictor = ZmqAdaptor(config=get_zmq_config()["predictor"]["sockets"], logger=get_logger())
# print(predictor.sockets)
# while True:
#     data = predictor.router_client.recv_multipart()
#     print("receive data:", pickle.loads(data[-1]))
#     send_data = {"action": random.randint(0, 1), "policy": [0.5, 0.5], "value": 1}
#     data[-1] = pickle.dumps(send_data)
#     print("send data:", send_data)
#     predictor.router_client.send_multipart(data)

predictor = Predictor()

predictor.run()
