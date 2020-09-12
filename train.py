import os
import shutil
import time
from collections import OrderedDict
import logging
from datetime import datetime

from tqdm import tqdm
import numpy as np
import hydra
import mlflow
from omegaconf import OmegaConf

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR

from src.losses import JointsMSELoss
from src.metrics import accuracy
from src.postprocess import get_final_preds
from src.optimizers import get_optimizer
from src.utils.utils import AverageMeter
from src.utils.debug import save_debug_images, save_checkpoint
from src.utils.mlflow import log_params_to_mlflow, search_run
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


def validate(config, val_loader, val_dataset, model, criterion, output_dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.model.num_joints, 3),
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
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            img_path.extend(meta['image'])

            idx += num_images

            if i % config.print_freq == 0:
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


@hydra.main(config_name='configs/config.yml')
def main(config):
    # Get original current directory path
    cwd = hydra.utils.get_original_cwd()

    # Get hydra current directory path
    hydra_dir = os.getcwd()

    # Set name
    if config.name is None:
        config.name = '%s_%s' % (
            config.model.name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        with open('.hydra/config.yaml', 'w') as file:
            file.write(OmegaConf.to_yaml(config))
    
    os.chdir(cwd)
    
    # Search mlflow run id
    run_id = search_run(config.name)
    if run_id is None:
        resume = False
    else:
        resume = True

    # Make model directory
    model_dir = os.path.join(cwd, 'models', config.name)
    if not resume:
        os.makedirs(model_dir, exist_ok=True)

    # Set tracking uri
    mlflow.set_tracking_uri('file:/' + os.path.join(cwd, 'mlruns'))

    # Start a MLflow run
    if resume:
        mlflow.start_run(run_id=run_id)
    else:
        mlflow.start_run(run_name=config.name)

    os.chdir(hydra_dir)

    # Save config files
    if not resume:
        shutil.copy('.hydra/config.yaml', model_dir)
        shutil.copy('.hydra/hydra.yaml', model_dir)
        shutil.copy('.hydra/overrides.yaml', model_dir)

    # Log hydra params
    if not resume:
        log_params_to_mlflow(config)

    # Create model
    model = getattr(models, config.model.name)(config)

    gpus = [int(i) for i in config.gpus.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    cudnn.benchmark = True

    # Get loss
    criterion = JointsMSELoss(
        use_target_weight=config.loss.use_target_weight).cuda()

    # Get optimizer
    optimizer = get_optimizer(config, model)

    # Get lr scheduler
    scheduler = MultiStepLR(
        optimizer,
        config.train.lr_step,
        config.train.lr_factor,
    )

    # Create dataset and data loader
    train_dataset = getattr(datasets, config.dataset.name)(
        config,
        config.dataset.root,
        config.dataset.train_set,
        is_train=True,
    )
    valid_dataset = getattr(datasets, config.dataset.name)(
        config,
        config.dataset.root,
        config.dataset.test_set,
        is_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size * len(gpus),
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.test.batch_size * len(gpus),
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True
    )

    start_epoch = 0
    best_ap = 0.0

    # Resume a training
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth.tar')
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_ap = checkpoint['ap']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    best_model = False
    for epoch in range(start_epoch, config.train.epochs):
        print('Epoch [%d/%d]' % (epoch + 1, config.train.epochs))

        # Train
        train_metrics = train(train_loader, model, criterion, optimizer)
        logger.info(train_metrics)

        # Log train metrics
        mlflow.log_metrics(train_metrics, step=epoch + 1)

        scheduler.step()

        # Evaluate on validation set
        val_metrics = validate(config, valid_loader, valid_dataset, model,
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
        }, best_model, model_dir)


if __name__ == '__main__':
    main()
