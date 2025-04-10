import os

from logging import getLogger, WARNING
getLogger('matplotlib').setLevel(WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import config
from phone import PAD_INDEX, phone_label_list

def plot(writer, lengths, inputs, targets, preds, tgamma, dgamma, logpo, logpd, lo_mean, lo_logvar, qo_mean, qo_logvar, po_mean, po_logvar):
    if writer.iteration % config.plot_iter_interval == 0:
        def to_phone(x):
            x = [phone_label_list[x] if x != PAD_INDEX else ' ' for x in x.detach().cpu().numpy().tolist()]
            x = [x[0] if isinstance(x, tuple) else x for x in x]
            y = []
            for c in x:
                y += [c]
                for _ in range(config.num_states_per_phoneme - 1):
                    y += [None]
            return y
        
        lengths_l, lengths_a = lengths
        
        nrow, ncol = 6, 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*12, nrow*10))
        idx = 0
        labels = to_phone(inputs[idx,:lengths_l[idx]])
        
        ax = axes[0][0]
        m = ax.imshow(targets[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)
        ax = axes[0][1]
        m = ax.imshow(preds[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)
        #
        ax = axes[1][0]
        m = ax.imshow(tgamma[idx,:lengths_l[idx],:lengths_a[idx]].detach().cpu().numpy(),
                      aspect='auto', vmin=0, vmax=1, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax = axes[1][1]
        logpo = logpo.masked_fill(logpo <= -1.0E+10, float('nan'))
        m = ax.imshow(logpo[idx,:lengths_l[idx],:lengths_a[idx]].detach().cpu().numpy(),
                      aspect='auto', origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        #
        ax = axes[2][0]
        m = ax.imshow(dgamma[idx,:lengths_l[idx],:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=0, vmax=1, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax = axes[2][1]
        pd = logpd.exp().masked_fill(logpd <= -1.0E+10, float('nan'))
        m = ax.imshow(pd[idx,:lengths_l[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=0, vmax=1, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)

        #
        ax = axes[3][0]
        m = ax.imshow(lo_mean[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)
        ax = axes[3][1]
        m = ax.imshow(lo_logvar[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)

        #
        ax = axes[4][0]
        m = ax.imshow(qo_mean[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)
        ax = axes[4][1]
        m = ax.imshow(qo_logvar[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)

        #
        ax = axes[5][0]
        m = ax.imshow(po_mean[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)
        ax = axes[5][1]
        m = ax.imshow(po_logvar[idx,:lengths_a[idx]].detach().cpu().numpy().T,
                      aspect='auto', vmin=-3, vmax=3, origin='lower', interpolation='nearest')
        fig.colorbar(m, ax=ax)
        
        plt.savefig(os.path.join(writer.tmpdir, 'check{}.pdf'.format(writer.iteration)), format='pdf', bbox_inches='tight')
        plt.close()
