import yaml


def load_yaml(file: str):
    with open(file, 'r', encoding='utf-8') as f:
        return yaml.full_load(f)
