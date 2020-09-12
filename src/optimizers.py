import torch.optim as optim


def get_optimizer(config, model):
    if config.train.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.lr,
            momentum=config.train.momentum,
            weight_decay=config.train.wd,
            nesterov=config.train.nesterov,
        )
    elif config.train.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.lr,
        )
    else:
        raise NotImplementedError('%s does not exist.' %
                                  config.train.optimizer)

    return optimizer
