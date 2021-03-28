from naive_rllib.utils import package_path, load_yaml
import os


def get_zmq_config():
    return load_yaml(os.path.join(package_path, "configs/zmq_setting.yaml"))


def get_agent_config():
    return load_yaml(os.path.join(package_path, "configs/agent_setting.yaml"))


print(get_agent_config())