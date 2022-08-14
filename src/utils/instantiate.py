import hydra


def instantiate(cfg):
    return hydra.utils.instantiate(cfg)