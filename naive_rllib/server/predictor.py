from naive_rllib.utils import ZmqAdaptor, get_logger
from naive_rllib.configs import get_zmq_config
import pickle
import random

predictor = ZmqAdaptor(config=get_zmq_config()["predictor"]["sockets"], logger=get_logger())
print(predictor.sockets)
while True:
    data = predictor.router_client.recv_multipart()
    print("receive data:", pickle.loads(data[-1]))
    send_data = {"action": random.randint(0,1), "policy": [0.5, 0.5], "value": 1}
    data[-1] = pickle.dumps(send_data)
    print("send data:", send_data)
    predictor.router_client.send_multipart(data)
