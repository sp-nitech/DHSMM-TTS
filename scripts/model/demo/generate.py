#!/usr/bin/env python3
import argparse
import numpy as np
import os
import random
import time
import torch

from train import build_model_and_optimizer
from phone import convert_to_phone_tensor
from utils import extend_seq
import config

def _generate(model, inputs):
    device = inputs.device
    input_lengths = torch.LongTensor([inputs.size(1)])
    
    def dur2attn(dur):
        B, L = dur.size()
        start_indices = torch.cumsum(torch.cat((torch.zeros(B, 1, dtype=torch.long, device=device), dur[:, :-1]), dim=-1), dim=-1)  # [B x L]
        end_indices = torch.cumsum(dur, dim=-1)  # [B x L]
        max_length = end_indices[:, -1].max().item()
        ids = torch.arange(0, max_length, dtype=torch.long, device=device).expand(B, L, max_length).transpose(1, 2)
        matrix = ((start_indices.unsqueeze(1) <= ids).long() * (ids < end_indices.unsqueeze(1)).long()).float()  # [B x T x L]
        return matrix
    
    po_s, ps, state_lengths = model.forward_mdn(inputs, input_lengths)
    
    dur = ps[0]
    dur = dur.round().clamp(min=1.0).long().squeeze(-1)
    attn = dur2attn(dur)

    o = po_s[0]  # [B x L x C]
    o = torch.bmm(attn, o)  # [B x T x C]
    px = model.speech_decoder(o, torch.LongTensor([o.size(1)]))
    x = px[0]
    out = extend_seq(x, config.mel_reduction_factor)
    return out, attn, dur

@torch.no_grad()
def generate(input_file, checkpoint_path, output_directory):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, _, _, _ = build_model_and_optimizer(checkpoint_path, device=device)
    model.eval()
    
    print(' Processing: {}'.format(input_file))
    base = os.path.basename(input_file)[:-len('.phn')]
    
    # read input data
    with open(input_file, 'r') as f: x = [line.strip().split(' ') for line in f.readlines()]
    x, y = list(zip(*x))
    inputs = convert_to_phone_tensor(x, y).unsqueeze(0).to(device)
    
    # forward
    start_time = time.perf_counter()
    out, align, dur = _generate(model, inputs)
    duration = time.perf_counter() - start_time
    print('  ({:>.4f} sec.)'.format(duration))
    
    outpath = os.path.join(output_directory, base)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    np.savez(outpath + '.npz', mel=out[0].detach().cpu().numpy())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('-o', '--output_directory', type=str, required=True, help='output directory path')
    parser.add_argument('-c', '--checkpoint_path', type=str, required=True, help='checkpoint path')
    args = parser.parse_args()
    
    print('=== Arguments ===')
    for key, value in vars(args).items():
        print('  {} = {}'.format(key, str(value)))
    print('')
    
    # set seed
    random_seed = 12345
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    generate(args.input_file, args.checkpoint_path, args.output_directory)
    
    print('\n\ndone')
