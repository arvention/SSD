from models.ssd import build_ssd
from models.fssd import build_fssd
from models.rfbnet import build_rfbnet


def get_model(config, anchors):
    """
    returns the model
    """

    model = None

    if config['model'] == 'SSD':
        model = build_ssd(mode=config['mode'],
                          new_size=config['new_size'],
                          anchors=anchors,
                          class_count=config['class_count'])

    elif config['model'] == 'FSSD':
        model = build_fssd(mode=config['mode'],
                           new_size=config['new_size'],
                           anchors=anchors,
                           class_count=config['class_count'])

    elif config['model'] == 'RFBNet':
        model = build_rfbnet(mode=config['mode'],
                             new_size=config['new_size'],
                             anchors=anchors,
                             class_count=config['class_count'])

    return model
