from naive_rllib.utils import ZmqAdaptor
from naive_rllib.configs import get_zmq_config
from naive_rllib.utils import get_logger
import os
import time


def test_create_socket():
    logger = get_logger()
    zmq_config = get_zmq_config()
    client_zmq = ZmqAdaptor(zmq_config["client"]["sockets"], logger=logger)
    assert client_zmq.sockets == ['req_predictor', 'push_trainer', 'push_logger']

    predictor_zmq = ZmqAdaptor(zmq_config["predictor"]["sockets"], logger=logger)
    assert predictor_zmq.sockets == ['router_client', 'sub_trainer', 'push_logger']

    trainer_zmq = ZmqAdaptor(zmq_config["trainer"]["sockets"], logger=logger)
    assert trainer_zmq.sockets == ['pull_client', 'pub_predictor', 'push_logger']

    logger_zmq = ZmqAdaptor(zmq_config["logger"]["sockets"], logger=logger)
    assert logger_zmq.sockets == ['pull_all']

    # while True:
    #     print("req发送：", "ABC")
    #     client_zmq.req_predictor.send("ABC".encode("utf-8"))
    #     a = predictor_zmq.router_client.recv_multipart()
    #     message = a[-1].decode()
    #     print("router收到:", message)
    #     a[-1] = message.lower().encode()
    #     predictor_zmq.router_client.send_multipart(a)
    #     b = client_zmq.req_predictor.recv().decode("utf-8")
    #     print("req收到响应：", b)
    #     print("=" * 50)
    #     time.sleep(5)


if __name__ == '__main__':
    test_create_socket()
