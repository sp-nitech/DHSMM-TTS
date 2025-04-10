import itertools
import torch

PAD_INDEX = 0
PAD = '[pad]'
BOS = '[bos]'
EOS = '[eos]'

phone_label_list = (PAD, 'sil', 'pau', ) + tuple(itertools.product([
    'N', 'a', 'b', 'by', 'ch', 'cl', 'd', 'dy', 'e', 'f',
    'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', 'm', 'my',
    'n', 'ny', 'o', 'p', 'py', 'r', 'ry', 's', 'sh', 't',
    'ts', 'ty', 'u', 'w', 'y', 'z',
], ['H', 'L']
))
assert phone_label_list.index(PAD) == PAD_INDEX

def convert_to_phone_tensor(phone_list, accent_list, msg=None):
    label_list = phone_label_list
    
    inputs = []
    for p, a in zip(phone_list, accent_list):
        x = p if a == 'xx' else (p, a, )
        if x in label_list:
            inputs.append(label_list.index(x))
        else:
            inputs.append(PAD_INDEX)
            s = "WARN: '{}' is not defined.".format(x)
            if msg is not None:
                s += ' ({})'.format(msg)
            print('WARN:', s)
        
    tensor = torch.LongTensor(inputs)
    return tensor
