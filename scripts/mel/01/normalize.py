import argparse
import numpy as np
import os

def calculate_stats(specdir, bases, ftype):
    ex = ex2 = None
    num_frames = 0
    for base in bases:
        specfile = os.path.join(specdir, base + '.npz')
        print('load {} to calculate variance'.format(specfile))
        x = np.load(specfile)[ftype].astype(np.float64)
        if ex is None:
            m = np.mean(x, axis=0)
            ex = np.sum(x - m, axis=0)
        else:
            ex += np.sum(x - m, axis=0)
        if ex2 is None:
            m2 = np.mean(x ** 2, axis=0)
            ex2 = np.sum(x ** 2 - m2, axis=0)
        else:
            ex2 += np.sum(x ** 2 - m2, axis=0)
        num_frames += x.shape[0]
    assert num_frames != 0
    ex = ex / num_frames + m
    ex2 = ex2 / num_frames + m2
    var = ex2 - ex ** 2
    return {'mean': ex.astype(np.float32),
            'var': var.astype(np.float32)}


def normalize(x, statsfile):
    narr = np.load(statsfile)
    mean, var = narr['mean'].astype(np.float64), narr['var'].astype(np.float64)
    std = np.sqrt(var)
    y = (x.astype(np.float64) - mean) / std
    return y.astype(np.float32)


def inverse(x, statsfile):
    narr = np.load(statsfile)
    mean, var = narr['mean'].astype(np.float64), narr['var'].astype(np.float64)
    std = np.sqrt(var)
    y = x.astype(np.float64) * std + mean
    return y.astype(np.float32)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scpfile', help='list file containing basenames of data')
    parser.add_argument('--specdir', default='spec')
    parser.add_argument('--ftype', default='mel', help='feature type')
    parser.add_argument('--calcstats', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    
    args = parser.parse_args()
    
    statsdir = 'stats'
    with open(args.scpfile, 'r') as f:
        bases = [os.path.join(*line.strip().split(' ')) for line in f.readlines()]
        
    if args.calcstats:
        os.makedirs(statsdir, exist_ok=True)
        statsfile = os.path.join(statsdir, 'stats_{}.npz'.format(args.ftype))
        np.savez(statsfile, **calculate_stats(args.specdir, bases, args.ftype))

    if args.normalize:
        for base in bases:
            print('normalizing for {}'.format(base))
            
            specfile = os.path.join(args.specdir, base + '.npz')
            specs = np.load(specfile)
            out = {}
            for ftype in specs.keys():
                statsfile = os.path.join(statsdir, 'stats_{}.npz'.format(ftype))
                if os.path.exists(statsfile):
                    out[ftype] = normalize(specs[ftype], statsfile)
                else:
                    out[ftype] = specs[ftype]
                    
            # save
            np.savez(specfile, **out)
            
