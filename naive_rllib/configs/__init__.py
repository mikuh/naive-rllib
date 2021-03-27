import yaml


def load_zmq_config(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        return yaml.full_load (f)