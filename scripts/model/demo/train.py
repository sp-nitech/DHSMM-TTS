#!/usr/bin/env python3
import argparse
import datetime
import math
import numpy as np
import os
import random
import sys
import time
import torch

from logging import getLogger, INFO, DEBUG, basicConfig
from torch.utils.data import DataLoader

from model.model import Model
from utils import Writer
from datasets import batch_to_tensors_and_lengths
from datasets import ATR_TTS_JP_CORPUS
import config


def train(args):
    logger = getLogger('setup')
    
    assert torch.cuda.is_available()
    device = torch.device('cuda:{:d}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
    assert not os.path.islink(args.output_directory), '{} is link.'.format(args.output_directory)
    os.makedirs(args.output_directory, exist_ok=True)
    
    logger.info('=== Arguments ===')
    for key, value in vars(args).items():
        logger.info('  {} = {}'.format(key, str(value)))
    logger.info('')
    
    logger.info('=== Model ===')
    model, optimizer, iteration, epoch = build_model_and_optimizer(args.checkpoint_path, device=device)
    logger.info(model)
    logger.info('Number of model parameters: {}'.format(
        sum([p.numel() for p in filter(lambda x: x.requires_grad, model.parameters())])))
    logger.info('')
    
    logger.info('=== Dataset ===')
    dataloader = prepare_dataloader(epoch)
    iter_per_epoch = len(dataloader)
    max_iteration = config.max_iteration
    max_epoch = int(math.floor(max_iteration / iter_per_epoch))
    logger.info('iter / epoch: {}'.format(iter_per_epoch))
    logger.info('max epoch: {}'.format(max_epoch))
    logger.info('max iteration: {}'.format(max_iteration))
    logger.info('')
    
    logger.info('=== Logger ===')
    logdir = os.path.join(args.output_directory, 'running_logs', datetime.datetime.now().strftime(u'%Y-%m-%d_%H:%M:%S'))
    writer = Writer(logdir, args.output_directory, iteration)
    logger.info('')

    logger = getLogger('train')
    logger.info('=== Start training ===')
    model.train()
    start_time = time.perf_counter()
    average_loss = 0.0
    average_norm = 0.0
    assert iteration < max_iteration
    try:
        while True:
            for batch in dataloader:
                if config.max_iteration <= iteration:
                    exit(0)
                
                # update temperature
                pw = min(1.0, max(0.0, (iteration // 5000 + 1) / 20.0)) ** 2
                
                # send to gpu
                ret = batch_to_tensors_and_lengths(batch)
                inputs, targets = tuple(x.to(device) for x in ret[0])
                lengths = ret[1]
                
                # reset
                model.zero_grad(set_to_none=True)
                
                # forward
                writer.logger.debug('Iteration {}'.format(iteration))
                loss = model(inputs, targets, lengths, pw, writer=writer)
                
                # nan check
                if loss != loss:
                    raise ValueError('ERROR: loss is NaN at iteration={}'.format(iteration))
                
                # backward
                loss.backward()
                
                loss = loss.item()
                average_loss += loss
                
                # clip gradient norm of parameters
                grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if grad_norm != grad_norm:
                    raise ValueError('ERROR: grad is {} at iteration={}'.format(grad_norm, iteration))
                average_norm += grad_norm
                
                if iteration % config.log_iter_interval == 0:
                    average_loss = average_loss / float(config.log_iter_interval)
                    average_norm = average_norm / float(config.log_iter_interval)
                    duration = (time.perf_counter() - start_time) / float(config.log_iter_interval)
                    logger.info('Iteration: {} loss: {:.6f} norm: {:.3e} {:.3f}sec'.format(
                        iteration, average_loss, average_norm, duration))
                    average_loss = 0
                    average_norm = 0
                    start_time = time.perf_counter()
                    
                if iteration % config.save_iter_interval == 0:
                    save_checkpoint(os.path.join(args.output_directory, 'checkpoint_{}'.format(iteration)), model, optimizer, iteration, epoch)
                
                # update
                if writer is not None:
                    writer.step()
                optimizer.step()
                iteration += 1
            else:
                epoch += 1
        # end while
    finally:
        save_checkpoint(os.path.join(args.output_directory, 'checkpoint_final'), model, optimizer, iteration, epoch)

def prepare_dataloader(epoch=0):
    key = 'atr-tts-jp-corpus'
    dataset = ATR_TTS_JP_CORPUS(key, config.dataset[key])
    loader = DataLoader(dataset,
                        num_workers=0,  # for cache
                        shuffle=True, sampler=None,
                        batch_size=config.batch_size,
                        pin_memory=True, drop_last=True,
                        collate_fn=lambda x: x)
    return loader


def save_checkpoint(filepath, model, optimizer, iteration, epoch):
    getLogger('setup').info('Saving model and optimizer state at iteration={}, epoch={} to {}'.format(iteration, epoch, filepath))
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'epoch': epoch,
    }, filepath)


def build_model_and_optimizer(checkpoint_path, device=torch.device('cuda:0')):
    iteration = 0
    epoch = 0
    
    if checkpoint_path is not None:
        assert os.path.isfile(checkpoint_path)
        getLogger('setup').info('Loading checkpoint "{}"'.format(checkpoint_path))
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), **config.optim)
    
    if checkpoint_path is not None:
        model.load_state_dict(checkpoint_dict['model'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        iteration = checkpoint_dict['iteration']
        epoch = checkpoint_dict['epoch']
    
    return model, optimizer, iteration, epoch

def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    grads = [p.grad for p in parameters if p.grad is not None]
    device = grads[0].device
    if norm_type == 'inf':
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    if max_norm is not None:
        coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for g in grads:
            g.detach().mul_(coef.to(g.device))
    return total_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, required=True,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, required=False,
                        help='checkpoint path to restart training')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    # set seed
    random_seed = 12345
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
    basicConfig(level=DEBUG if args.debug else INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    train(args)
    
