import numpy as np
import os
import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import reduce_seq
from phone import PAD_INDEX
from phone import convert_to_phone_tensor
import config

def get_lengths(x):
    return torch.LongTensor([len(y) for y in x])

def batch_to_tensors_and_lengths(batch):
    phone, mel, = list(zip(*batch))
    lengths = (get_lengths(phone), get_lengths(mel), )
    phone = pad_sequence(phone, batch_first=True, padding_value=PAD_INDEX)
    mel = pad_sequence(mel, batch_first=True)
    return (phone, mel, ), lengths

class ATR_TTS_JP_CORPUS(Dataset):
    def __init__(self, name, conf):
        self.name = name
        with open(conf['datalist'], 'r') as f:
            bases = [os.path.join(*line.strip().split(' ')) for line in f.readlines()]
        self.bases = sorted(bases)
        self.dirs = conf['datadirs']
        self.cache = [None] * len(self.bases)
        
    def __getitem__(self, idx):
        if self.cache[idx] is None:
            phone = self.get_phone(os.path.join(self.dirs['phone'], self.bases[idx] + '.phn'))
            mel = self.get_mel(os.path.join(self.dirs['mel'], self.bases[idx] + '.npz'))
            self.cache[idx] = (phone, mel)
        return self.cache[idx]
    
    def get_phone(self, phnfile):
        with open(phnfile, 'r') as f: x = [line.strip().split(' ') for line in f.readlines()]
        x, y = list(zip(*x))
        phone = convert_to_phone_tensor(x, y, msg=phnfile)
        return phone
        
    def get_mel(self, specfile):
        mel = torch.from_numpy(np.load(specfile)['mel'])
        mel = reduce_seq(mel, factor=config.mel_reduction_factor)
        return mel
    
    def __len__(self):
        return len(self.bases)
