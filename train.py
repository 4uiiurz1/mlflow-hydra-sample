"""
Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
Original licence: Copyright (c) Microsoft, under the MIT License.
"""

import os
import time
from collections import OrderedDict
import logging
from datetime import datetime

from tqdm import tqdm
import numpy as np
import hydra
import mlflow

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from src.losses import JointsMSELoss
from src.metrics import accuracy
from src.postprocess import get_final_preds
from src.utils.utils import AverageMeter
from src.utils.debug import save_debug_images, save_checkpoint
from src.utils.mlflow import log_params_from_omegaconf_dict, start_run
import src.models as models
import src.datasets as datasets


logger = logging.getLogger(__name__)


def train(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    pbar = tqdm(total=len(train_loader))

    end = time.time()
    for input, target, target_weight, _ in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, _ = accuracy(
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
        )
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        postfix = OrderedDict([
            ('batch_time', batch_time.avg),
            ('loss', losses.avg),
            ('acc', acc.avg),
        ])

        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return {
        'train_batch_time': batch_time.avg,
        'train_loss': losses.avg,
        'train_acc': acc.avg,
    }


def validate(cfg, val_loader, val_dataset, model, criterion, output_dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, cfg.model.num_joints, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    img_path = []
    idx = 0

    pbar = tqdm(total=len(val_loader))

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            output = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            img_path.extend(meta['image'])

            idx += num_images

            if i % cfg.print_freq == 0:
                prefix = '{}_{}'.format('val', i)
                save_debug_images(input, meta, target, pred * 4, output,
                                  prefix)

            postfix = OrderedDict([
                ('batch_time', batch_time.avg),
                ('loss', losses.avg),
                ('acc', acc.avg),
            ])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        metrics, _ = val_dataset.evaluate(
            all_preds, output_dir, all_boxes, img_path)

    return {
        'val_batch_time': batch_time.avg,
        'val_loss': losses.avg,
        'val_acc': acc.avg,
        'ap': metrics['AP'],
        'ap_50': metrics['Ap .5'],
        'ap_75': metrics['AP .75'],
        'ap_m': metrics['AP (M)'],
        'ap_l': metrics['AP (L)'],
        'ar': metrics['AR'],
        'ar_50': metrics['AR .5'],
        'ar_75': metrics['AR .75'],
        'ar_m': metrics['AR (M)'],
        'ar_l': metrics['AR (L)'],
    }


def get_optimizer(cfg, model):
    if cfg.train.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum,
            weight_decay=cfg.train.wd,
            nesterov=cfg.train.nesterov,
        )
    elif cfg.train.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
        )
    else:
        raise NotImplementedError

    return optimizer


@hydra.main(config_name='configs/config.yml')
def main(cfg):
    # Set run name
    run_name = '%s_%s' % (
        cfg.model.name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    # Get original current directory path
    cwd = hydra.utils.get_original_cwd()

    # Set tracking uri
    mlflow.set_tracking_uri('file:/' + os.path.join(cwd, 'mlruns'))

    # Start a new MLflow run
    mlflow.start_run(run_name=run_name)

    # Log hydra params
    log_params_from_omegaconf_dict(cfg)

    # Create model
    model = getattr(models, cfg.model.name)(cfg)

    gpus = [int(i) for i in cfg.gpus.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    cudnn.benchmark = True

    # Get loss
    criterion = JointsMSELoss(
        use_target_weight=cfg.loss.use_target_weight).cuda()

    # Get optimizer
    optimizer = get_optimizer(cfg, model)

    # Get lr scheduler
    scheduler = MultiStepLR(
        optimizer,
        cfg.train.lr_step,
        cfg.train.lr_factor,
    )

    # Create dataset and data loader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = getattr(datasets, cfg.dataset.name)(
        cfg,
        cfg.dataset.root,
        cfg.dataset.train_set,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = getattr(datasets, cfg.dataset.name)(
        cfg,
        cfg.dataset.root,
        cfg.dataset.test_set,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size * len(gpus),
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.test.batch_size * len(gpus),
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True
    )

    best_ap = 0.0
    best_model = False
    for epoch in range(0, cfg.train.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, cfg.train.epochs))

        # Train
        train_metrics = train(train_loader, model, criterion, optimizer)
        logger.info(train_metrics)

        # Log train metrics
        mlflow.log_metrics(train_metrics, step=epoch + 1)

        scheduler.step()

        # Evaluate on validation set
        val_metrics = validate(cfg, valid_loader, valid_dataset, model,
                               criterion, '.')
        logger.info(val_metrics)

        # Log val metrics
        mlflow.log_metrics(val_metrics, step=epoch + 1)

        if val_metrics['ap'] > best_ap:
            best_ap = val_metrics['ap']
            best_model = True
        else:
            best_model = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ap': val_metrics['ap'],
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, best_model, os.path.join(cwd, 'models', cfg.model.name))


if __name__ == '__main__':
    main()
