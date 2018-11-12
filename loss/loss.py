from loss.multibox_loss import MultiBoxLoss


def get_loss(config):
    """
    returns the loss function
    """

    loss = None

    if config['loss_config'] == 'multibox':
        loss = MultiBoxLoss(class_count=config['class_count'],
                            threshold=config['threshold'],
                            pos_neg_ratio=config['pos_neg_ratio'],
                            use_gpu=config['use_gpu'])

    return loss
