from arch.ssd import build_ssd
from arch.fssd import build_fssd


def get_arch(config):
    """
    returns the architecture
    """

    arch = None

    if config['architecture'] == 'SSD':
        arch = build_ssd(mode=config['mode'],
                         new_size=config['new_size'],
                         class_count=config['class_count'])

    elif config['architecture'] == 'FSSD':
        arch = build_fssd(mode=config['mode'],
                          new_size=config['new_size'],
                          class_count=config['class_count'])

    return arch
