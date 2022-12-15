import yaml


def code_from_str(str):
    pass


if __name__ == '__main__':
    file_path = './config/config.yaml'
    with open(file_path, encoding='utf-8') as f:
        config = yaml.load(f)
        print(config)

