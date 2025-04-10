from logging import getLogger
import os
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

@torch.jit.script
def reduce_seq(x: Tensor, factor: int) -> Tensor:
    shape = list(x.size())
    T, C = shape[-2:]
    RT = T // factor
    if RT * factor != T: x = x[..., :RT * factor, :]
    rx = x.reshape(shape[:-2] + [RT, C * factor])
    return rx

@torch.jit.script
def extend_seq(rx: Tensor, factor: int) -> Tensor:
    shape = list(rx.size())
    RT, RC = shape[-2:]
    x = rx.reshape(shape[:-2] + [RT * factor, RC // factor])
    return x

class Writer:
    def __init__(self, logdir, tmpdir, iteration):
        self.logger = getLogger('writer')
        self.iteration = iteration
        self.tmpdir = tmpdir
        self.logger.info('Save directory location: {}'.format(logdir))
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)
        
    def __call__(self, name, loss):
        with torch.no_grad():
            if isinstance(loss, float) or 0 == loss.ndim:
                v = loss
            else:
                v = loss.detach().mean().item()
            self.writer.add_scalar(name, v, self.iteration)
            self.logger.debug('{}: {}'.format(name, v))
    
    def step(self):
        self.iteration += 1
    
